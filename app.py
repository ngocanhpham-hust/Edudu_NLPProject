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
from rank_bm25 import BM25Okapi

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from study_assistant.config import settings
from study_assistant.document import build_chunks_from_pdfs
from study_assistant.generation import (
    GeminiClient,
    answer_question,
    generate_flashcards,
    generate_quiz,
    retrieved_from_chunks,
    result_to_jsonable,
    summarize_document,
    summarize_topic,
)
from study_assistant.retrieval import StudyIndex, tokenize
from study_assistant.schemas import RetrievedChunk


NO_DOCUMENT_ANSWER = "Không tìm thấy câu trả lời từ tài liệu đã tải lên."


class LexicalFallbackIndex:
    """BM25-only fallback used when the embedding model cannot be loaded."""

    def __init__(self, pdf_paths: list[str]) -> None:
        self.chunks = build_chunks_from_pdfs(
            pdf_paths,
            settings.chunk_size_words,
            settings.chunk_overlap_words,
            strategy="semantic",
        )
        if not self.chunks:
            raise ValueError("No chunks found in uploaded documents.")
        self.bm25 = BM25Okapi([tokenize(chunk.text, settings.bm25_tokenizer) for chunk in self.chunks])

    def retrieve_final(self, query: str, k: int = settings.final_top_k, **_kwargs):
        return self.retrieve(query, k=k)

    def retrieve(self, query: str, k: int = settings.final_top_k, **_kwargs):
        query_tokens = tokenize(query, settings.bm25_tokenizer)
        if not query_tokens:
            return []
        scores = self.bm25.get_scores(query_tokens)
        ranked_ids = sorted(range(len(scores)), key=lambda idx: float(scores[idx]), reverse=True)[:k]
        if not ranked_ids or float(scores[ranked_ids[0]]) <= 0.0:
            return []
        results = []
        for rank, idx in enumerate(ranked_ids, start=1):
            score = float(scores[idx])
            if score <= 0.0:
                continue
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=score,
                    source_marker=f"S{rank}",
                    bm25_score=score,
                )
            )
        return results


@st.cache_resource(show_spinner=False)
def build_index_from_paths(
    paths: tuple[str, ...],
) -> StudyIndex | LexicalFallbackIndex:
    index = StudyIndex(
        embedding_model_name=settings.embedding_model,
        bm25_tokenizer=settings.bm25_tokenizer,
    )
    try:
        index.build_from_pdfs(
            list(paths),
            chunk_size_words=settings.chunk_size_words,
            chunk_overlap_words=settings.chunk_overlap_words,
            chunking_strategy="semantic",
        )
        return index
    except Exception:
        return LexicalFallbackIndex(list(paths))


def retrieve_chunks(index: StudyIndex, query: str, k: int = settings.final_top_k):
    try:
        return index.retrieve_final(
            query,
            k=k,
            alpha=settings.hybrid_alpha,
            candidate_k=max(settings.candidate_k, settings.rerank_initial_k),
            rerank_initial_k=settings.rerank_initial_k,
        )
    except Exception:
        try:
            return index.retrieve(
                query,
                method="hybrid",
                k=k,
                alpha=settings.hybrid_alpha,
                candidate_k=settings.candidate_k,
            )
        except Exception:
            return []


def has_no_document_answer(answer: str) -> bool:
    normalized = answer.lower()
    no_answer_phrases = [
        "document does not provide enough information",
        "provided information is insufficient",
        "không cung cấp đủ thông tin",
        "không có đủ thông tin",
        "không tìm thấy",
        "không đề cập",
        "không có câu trả lời",
    ]
    return any(phrase in normalized for phrase in no_answer_phrases)


def show_no_document_answer() -> None:
    st.info(NO_DOCUMENT_ANSWER)


def show_generation_error() -> None:
    st.info("Không thể tạo câu trả lời. Vui lòng kiểm tra API key và thử lại.")


def require_document_chunks(chunks) -> bool:
    if chunks:
        return True
    show_no_document_answer()
    return False


def show_sources(chunks):
    st.subheader("Sources")
    for c in chunks:
        page_start = c.chunk.page_start or c.chunk.page
        page_end = c.chunk.page_end or c.chunk.page
        page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        with st.expander(f"{c.source_marker} - {c.chunk.filename} - {page_label}"):
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
    st.set_page_config(page_title="Edudu - Smart Study Assistant", layout="wide")
    st.title("Edudu - Smart Study Assistant")

    with st.sidebar:
        st.header("Files")
        uploaded = st.file_uploader("Import files", type=["pdf"], accept_multiple_files=True)
        st.header("API key")
        api_key = st.text_input("Google API key", value=os.getenv("GOOGLE_API_KEY", ""), type="password")

    if not uploaded:
        return

    paths = save_uploads(uploaded)
    with st.spinner("Preparing study materials..."):
        try:
            index = build_index_from_paths(tuple(str(p) for p in paths))
        except Exception:
            st.info("Không thể đọc tài liệu đã tải lên. Vui lòng thử lại với file PDF khác.")
            return

    client = GeminiClient(api_key=api_key) if api_key else None
    tabs = st.tabs(["Q&A", "Summarize", "Quiz", "Flashcards"])

    with tabs[0]:
        question = st.text_area("Question", value="What is the main idea of this document?", height=130)
        if st.button("Answer", key="ask"):
            chunks = retrieve_chunks(index, question)
            if not require_document_chunks(chunks):
                pass
            elif not api_key:
                st.warning("Add API key to generate an answer.")
                show_sources(chunks)
            else:
                with st.spinner("Thinking..."):
                    try:
                        result = answer_question(question, chunks, client)
                    except Exception:
                        show_generation_error()
                    else:
                        if has_no_document_answer(result.answer):
                            show_no_document_answer()
                        else:
                            st.markdown(result.answer)
                            for warning in result.warnings:
                                st.warning(warning)
                            show_sources(result.chunks)

    with tabs[1]:
        topic = st.text_input("Summary topic", value="main concepts")
        full_document = st.checkbox("Full document", value=False)
        if st.button("Summarize", key="summary"):
            if full_document:
                chunks = retrieved_from_chunks(index.chunks)
            else:
                chunks = retrieve_chunks(index, topic, settings.summary_retrieval_k)
            if not require_document_chunks(chunks):
                pass
            elif not api_key:
                st.warning("Add API key to generate a summary.")
                show_sources(chunks)
            else:
                try:
                    result = summarize_document(topic, chunks, client) if full_document else summarize_topic(topic, chunks, client)
                except Exception:
                    show_generation_error()
                else:
                    if has_no_document_answer(result.summary):
                        show_no_document_answer()
                    else:
                        st.subheader("Summary")
                        st.write(result.summary)
                        st.subheader("Highlights")
                        for p in result.key_points:
                            st.write("-", p)
                        st.download_button("Download JSON", json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2), "summary.json")
                        show_sources(result.chunks)

    with tabs[2]:
        topic = st.text_input("Quiz topic", value="main concepts", key="quiz_topic")
        count = st.number_input("Number of questions", min_value=1, max_value=20, value=settings.quiz_default_count)
        if st.button("Generate quiz", key="quiz"):
            chunks = retrieve_chunks(index, topic)
            if not require_document_chunks(chunks):
                pass
            elif not api_key:
                st.warning("Add API key to generate quiz.")
                show_sources(chunks)
            else:
                try:
                    result = generate_quiz(topic, chunks, int(count), client)
                except Exception:
                    show_generation_error()
                else:
                    if not result.items:
                        show_no_document_answer()
                    else:
                        for i, item in enumerate(result.items, start=1):
                            st.markdown(f"**Q{i}. {item.question}**")
                            for j, opt in enumerate(item.options):
                                prefix = "Correct:" if j == item.correct_index else "-"
                                st.write(prefix, opt)
                            st.caption(item.explanation)
                        st.download_button("Download JSON", json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2), "quiz.json")
                        show_sources(result.chunks)

    with tabs[3]:
        topic = st.text_input("Flashcard topic", value="main concepts", key="flash_topic")
        count = st.number_input("Number of flashcards", min_value=1, max_value=30, value=settings.flashcard_default_count)
        if st.button("Generate flashcards", key="flash"):
            chunks = retrieve_chunks(index, topic)
            if not require_document_chunks(chunks):
                pass
            elif not api_key:
                st.warning("Add API key to generate flashcards.")
                show_sources(chunks)
            else:
                try:
                    result = generate_flashcards(topic, chunks, int(count), client)
                except Exception:
                    show_generation_error()
                else:
                    if not result.cards:
                        show_no_document_answer()
                    else:
                        for i, card in enumerate(result.cards, start=1):
                            with st.expander(f"Card {i}: {card.front}"):
                                st.write(card.back)
                                if card.hint:
                                    st.caption(f"Hint: {card.hint}")
                        st.download_button("Download JSON", json.dumps(result_to_jsonable(result), ensure_ascii=False, indent=2), "flashcards.json")
                        show_sources(result.chunks)


if __name__ == "__main__":
    main()
