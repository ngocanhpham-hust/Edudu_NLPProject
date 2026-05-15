import json

import httpx
import streamlit as st

from src.config import settings
from src.interfaces.styles import GLOBAL_CSS

_API = settings.api_url


def _api(method: str, path: str, **kwargs):
    response = httpx.request(method, f"{_API}{path}", timeout=180, **kwargs)
    response.raise_for_status()
    return response.json()


def _sidebar():
    with st.sidebar:
        st.header("Tài liệu")
        uploaded = st.file_uploader("Upload PDF", type=["pdf"])
        if uploaded and st.button("Upload và index"):
            files = {"file": (uploaded.name, uploaded.getvalue(), "application/pdf")}
            st.success(_api("POST", "/upload", files=files))

        if st.button("Index lại thư mục data/"):
            st.info(_api("POST", "/ingest", params={"recreate": False}))

        docs = _api("GET", "/documents")
        filenames = [doc["filename"] for doc in docs]
        selected = st.multiselect("Phạm vi file", filenames)
        page = st.number_input("Trang", min_value=0, value=0)
        return selected, page or None


def _filters(filenames, page):
    payload = {}
    if filenames:
        payload["filenames" if len(filenames) > 1 else "filename"] = filenames if len(filenames) > 1 else filenames[0]
    if page and len(filenames) <= 1:
        payload["page"] = int(page)
    return payload or None


def _show_sources(result):
    citations = result.get("citations") or []
    if citations:
        st.caption("Nguồn")
        st.markdown(
            " ".join(
                f'<span class="source-chip">[{c["source_marker"]}] {c["filename"]}, trang {c["page"]}</span>'
                for c in citations
            ),
            unsafe_allow_html=True,
        )


def _tab_chat(filenames, page):
    question = st.text_area("Câu hỏi", height=110)
    k = st.slider("Số chunk", 1, 20, settings.top_k)
    if st.button("Hỏi", type="primary") and question.strip():
        result = _api("POST", "/ask", json={"question": question, "k": k, "filters": _filters(filenames, page)})
        st.markdown(result["answer"])
        _show_sources(result)


def _tab_summary(filenames, page):
    query = st.text_input("Tóm tắt theo truy vấn")
    if st.button("Tạo tóm tắt", type="primary"):
        result = _api("POST", "/summarize", json={"query": query or None, "filters": _filters(filenames, page)})
        st.markdown(result["summary"])
        for point in result.get("key_points", []):
            st.markdown(f"- {point}")
        _show_sources(result)
        st.download_button("Tải JSON", json.dumps(result, ensure_ascii=False, indent=2), "summary.json")


def _tab_quiz(filenames, page):
    query = st.text_input("Chủ đề quiz")
    count = st.number_input("Số câu", min_value=1, max_value=50, value=settings.quiz_default_count)
    if st.button("Tạo quiz", type="primary"):
        result = _api("POST", "/quiz", json={"query": query or None, "count": count, "filters": _filters(filenames, page)})
        for idx, item in enumerate(result.get("items", []), 1):
            st.subheader(f"Câu {idx}. {item['question']}")
            st.radio("Lựa chọn", item["options"], key=f"quiz-{idx}", index=None)
            st.caption(f"Đáp án: {item['options'][item['correct_index']]}")
            st.write(item["explanation"])
        _show_sources(result)


def _tab_flashcards(filenames, page):
    query = st.text_input("Chủ đề flashcards")
    count = st.number_input("Số thẻ", min_value=1, max_value=100, value=settings.flashcards_default_count)
    if st.button("Tạo flashcards", type="primary"):
        result = _api("POST", "/flashcards", json={"query": query or None, "count": count, "filters": _filters(filenames, page)})
        for card in result.get("cards", []):
            with st.expander(card["front"]):
                st.write(card["back"])
                if card.get("hint"):
                    st.caption(card["hint"])
        _show_sources(result)


def run():
    st.set_page_config(page_title="Simple NotebookLM", layout="wide")
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.title("Simple NotebookLM")
    filenames, page = _sidebar()
    tabs = st.tabs(["Hỏi đáp", "Tóm tắt", "Quiz", "Flashcards"])
    for tab, fn in zip(tabs, [_tab_chat, _tab_summary, _tab_quiz, _tab_flashcards]):
        with tab:
            fn(filenames, page)


if __name__ == "__main__":
    run()
