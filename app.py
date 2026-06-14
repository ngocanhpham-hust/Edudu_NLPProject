"""Local Streamlit demo.

Run locally:
    streamlit run app.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from study_assistant.config import settings
from study_assistant.generation import (
    GeminiClient,
    answer_question,
    extract_information,
    generate_flashcards,
    generate_quiz,
    retrieved_from_chunks,
    result_to_jsonable,
    summarize_document,
    summarize_topic,
)
from study_assistant.retrieval import StudyIndex


@st.cache_resource(show_spinner=False)
def build_index_from_paths(
    paths: tuple[str, ...],
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    bm25_tokenizer: str,
    chunking_strategy: str,
) -> StudyIndex:
    index = StudyIndex(embedding_model_name=embedding_model, bm25_tokenizer=bm25_tokenizer)
    index.build_from_pdfs(
        list(paths),
        chunk_size_words=chunk_size,
        chunk_overlap_words=overlap,
        chunking_strategy=chunking_strategy,
    )
    return index


def retrieve_chunks(index: StudyIndex, query: str, method: str, k: int, alpha: float, candidate_k: int, rerank_initial_k: int):
    if method == "final":
        return index.retrieve_final(
            query,
            k=k,
            alpha=alpha,
            candidate_k=max(candidate_k, rerank_initial_k),
            rerank_initial_k=rerank_initial_k,
        )
    return index.retrieve(query, method=method, k=k, alpha=alpha, candidate_k=candidate_k)


def show_sources(chunks):
    st.subheader("Retrieved sources")
    for c in chunks:
        page_start = c.chunk.page_start or c.chunk.page
        page_end = c.chunk.page_end or c.chunk.page
        page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        with st.expander(f"{c.source_marker} | {c.chunk.filename} | {page_label} | score={c.score:.4f}"):
            st.write(c.text)


def save_uploads(uploaded_files) -> list[Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="study_assistant_uploads_"))
    paths: list[Path] = []
    for file in uploaded_files:
        dest = tmp_dir / Path(file.name).name
        dest.write_bytes(file.read())
        paths.append(dest)
    return paths


def main():
    st.set_page_config(page_title="🤓 Edudu Study Assistant", layout="wide")
    st.title("Lightweight NotebookLM-inspired NLP Study Assistant")

    with st.sidebar:
        st.header("1. Upload PDFs")
        uploaded = st.file_uploader("Upload text-based PDF files", type=["pdf"], accept_multiple_files=True)
        st.header("2. Retrieval settings")
        chunk_size = st.number_input("chunk_size_words", min_value=100, max_value=2000, value=settings.chunk_size_words, step=50)
        overlap = st.number_input("chunk_overlap_words", min_value=0, max_value=500, value=settings.chunk_overlap_words, step=10)
        chunking_strategy = st.selectbox("chunking strategy", ["semantic", "paragraph", "naive"], index=0)
        retrieval_method = st.selectbox("retrieval method", ["final", "hybrid", "bm25", "dense"])
        alpha = st.slider("hybrid alpha", min_value=0.0, max_value=1.0, value=settings.hybrid_alpha, step=0.05)
        candidate_k = st.number_input("candidate_k", min_value=5, max_value=100, value=settings.candidate_k, step=5)
        rerank_initial_k = st.number_input("rerank_initial_k", min_value=5, max_value=100, value=settings.rerank_initial_k, step=5)
        top_k = st.number_input("top_k", min_value=1, max_value=15, value=settings.final_top_k, step=1)
        embedding_model = st.text_input("Embedding model", value=settings.embedding_model)
        bm25_tokenizer = st.selectbox("BM25 tokenizer", ["regex", "simple", "vi", "vietnamese"], index=0)
        st.header("3. Gemini")
        api_key = st.text_input("GOOGLE_API_KEY", value=os.getenv("GOOGLE_API_KEY", ""), type="password")

    if not uploaded:
        st.info("Upload at least one PDF to start.")
        return

    paths = save_uploads(uploaded)
    with st.spinner("Processing..."):
        index = build_index_from_paths(
            tuple(str(p) for p in paths),
            int(chunk_size),
            int(overlap),
            embedding_model,
            bm25_tokenizer,
            chunking_strategy,
        )
    st.success(f"Indexed {len(index.chunks)} chunks from {len(paths)} PDF file(s).")

    client = GeminiClient(api_key=api_key) if api_key else None
    tabs = st.tabs(["Q&A", "Extract", "Summarize", "Quiz", "Flashcards", "Debug retrieval"])

    with tabs[0]:
        question = st.text_area("Question", value="What is the main idea of this document?")
        if st.button("Answer", key="ask"):
            chunks = retrieve_chunks(index, question, retrieval_method, int(top_k), float(alpha), int(candidate_k), int(rerank_initial_k))
            if not api_key:
                st.warning("Set GOOGLE_API_KEY to generate an answer.")
                show_sources(chunks)
            else:
                with st.spinner("Thinking..."):
                    result = answer_question(question, chunks, client)
                st.markdown(result.answer)
                for warning in result.warnings:
                    st.warning(warning)
                show_sources(result.chunks)

    with tabs[1]:
        request = st.text_area("Extraction request", value="Extract key definitions and important formulas.")
        if st.button("Extract", key="extract"):
            chunks = retrieve_chunks(index, request, retrieval_method, int(top_k), float(alpha), int(candidate_k), int(rerank_initial_k))
            if not api_key:
                st.warning("Set GOOGLE_API_KEY to generate extracted information.")
                show_sources(chunks)
            else:
                result = extract_information(request, chunks, client)
                st.markdown(result.answer)
                for warning in result.warnings:
                    st.warning(warning)
                show_sources(result.chunks)

    with tabs[2]:
        topic = st.text_input("Summary topic", value="main concepts")
        full_document = st.checkbox("Summarize full document", value=False)
        if st.button("Summarize", key="summary"):
            if full_document:
                chunks = retrieved_from_chunks(index.chunks)
            else:
                chunks = retrieve_chunks(index, topic, retrieval_method, settings.summary_retrieval_k, float(alpha), int(candidate_k), int(rerank_initial_k))
            if not api_key:
                st.warning("Set GOOGLE_API_KEY to generate a summary.")
                show_sources(chunks)
            else:
                result = summarize_document(topic, chunks, client) if full_document else summarize_topic(topic, chunks, client)
                st.subheader("Summary")
                st.write(result.summary)
                st.subheader("Key points")
                for p in result.key_points:
                    st.write("-", p)
                st.download_button("Download JSON", json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2), "summary.json")
                show_sources(result.chunks)

    with tabs[3]:
        topic = st.text_input("Quiz topic", value="main concepts", key="quiz_topic")
        count = st.number_input("Number of questions", min_value=1, max_value=20, value=settings.quiz_default_count)
        if st.button("Generate quiz", key="quiz"):
            chunks = retrieve_chunks(index, topic, retrieval_method, int(top_k), float(alpha), int(candidate_k), int(rerank_initial_k))
            if not api_key:
                st.warning("Set GOOGLE_API_KEY to generate quiz.")
                show_sources(chunks)
            else:
                result = generate_quiz(topic, chunks, int(count), client)
                for i, item in enumerate(result.items, start=1):
                    st.markdown(f"**Q{i}. {item.question}**")
                    for j, opt in enumerate(item.options):
                        prefix = "✅" if j == item.correct_index else "○"
                        st.write(prefix, opt)
                    st.caption(item.explanation)
                st.download_button("Download JSON", json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2), "quiz.json")
                show_sources(result.chunks)

    with tabs[4]:
        topic = st.text_input("Flashcard topic", value="main concepts", key="flash_topic")
        count = st.number_input("Number of flashcards", min_value=1, max_value=30, value=settings.flashcard_default_count)
        if st.button("Generate flashcards", key="flash"):
            chunks = retrieve_chunks(index, topic, retrieval_method, int(top_k), float(alpha), int(candidate_k), int(rerank_initial_k))
            if not api_key:
                st.warning("Set GOOGLE_API_KEY to generate flashcards.")
                show_sources(chunks)
            else:
                result = generate_flashcards(topic, chunks, int(count), client)
                for i, card in enumerate(result.cards, start=1):
                    with st.expander(f"Card {i}: {card.front}"):
                        st.write(card.back)
                        if card.hint:
                            st.caption(f"Hint: {card.hint}")
                st.download_button("Download JSON", json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2), "flashcards.json")
                show_sources(result.chunks)

    with tabs[5]:
        query = st.text_input("Debug query", value="RAG pipeline")
        method = st.selectbox("Method", ["bm25", "dense", "hybrid", "rrf"])
        if st.button("Retrieve", key="debug"):
            chunks = index.retrieve(query, method=method, k=int(top_k), alpha=float(alpha), candidate_k=int(candidate_k))
            show_sources(chunks)


if __name__ == "__main__":
    main()
