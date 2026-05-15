from functools import lru_cache
from pathlib import Path

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from src.config import settings
from src.filters import filters_to_qdrant
from src.llm import invoke_llm
from src.offline import answer_from_chunks, keyword_retrieve, select_representative_chunks
from src.schemas import ChunkMetadata, Citation, RagAnswer, RetrievedChunk
from src.store import get_vector_store, scroll_all

PROMPTS_DIR = Path(__file__).parent / "prompts"
ANSWER_TEMPLATE = "answer.jinja2"


def retrieve(
    query: str,
    k: int | None = None,
    filters: dict | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    if settings.llm_provider == "echo":
        return keyword_retrieve(
            query,
            fetch_all_chunks(filters=filters, collection_name=collection_name),
            limit=k or settings.top_k,
        )

    try:
        hits = get_vector_store(collection_name).similarity_search_with_score(
            query=query,
            k=k or settings.top_k,
            filter=filters_to_qdrant(filters),
        )
    except Exception:
        if settings.llm_provider != "echo":
            raise
        return keyword_retrieve(
            query,
            fetch_all_chunks(filters=filters, collection_name=collection_name),
            limit=k or settings.top_k,
        )
    return [
        RetrievedChunk(
            text=doc.page_content,
            score=float(score),
            metadata=ChunkMetadata(**doc.metadata),
        )
        for doc, score in hits
    ]


def fetch_all_chunks(
    filters: dict | None = None,
    collection_name: str | None = None,
) -> list[RetrievedChunk]:
    name = collection_name or settings.qdrant_collection
    results: list[RetrievedChunk] = []
    for page in scroll_all(name, scroll_filter=filters_to_qdrant(filters)):
        for point in page:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            text = payload.get("page_content") or ""
            if meta and text:
                results.append(
                    RetrievedChunk(text=text, score=0.0, metadata=ChunkMetadata(**meta))
                )
    return sorted(
        results,
        key=lambda r: (
            r.metadata.filename,
            r.metadata.page,
            int(r.metadata.chunk_id.rsplit(":", 1)[-1]),
        ),
    )


@lru_cache(maxsize=1)
def _jinja_env() -> Environment:
    return Environment(
        loader=FileSystemLoader(str(PROMPTS_DIR)),
        autoescape=False,
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_prompt(template_name: str, **context) -> str:
    return _jinja_env().get_template(template_name).render(**context)


def format_citations(chunks: list[RetrievedChunk]) -> list[Citation]:
    return [
        Citation(
            source_index=i,
            source_marker=f"S{i}",
            filename=c.metadata.filename,
            page=c.metadata.page,
            section=c.metadata.section,
            chunk_id=c.metadata.chunk_id,
        )
        for i, c in enumerate(chunks, start=1)
    ]


def answer(
    question: str,
    k: int | None = None,
    filters: dict | None = None,
    collection_name: str | None = None,
) -> RagAnswer:
    broad_question = any(
        phrase in question.lower()
        for phrase in [
            "tóm tắt",
            "tom tat",
            "nội dung chính",
            "noi dung chinh",
            "chủ đề chính",
            "chu de chinh",
            "tài liệu này",
            "tai lieu nay",
        ]
    )
    if settings.llm_provider == "echo" and broad_question and filters:
        chunks = select_representative_chunks(
            fetch_all_chunks(filters=filters, collection_name=collection_name), limit=k or 18
        )
    else:
        chunks = retrieve(question, k=k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.",
        )

    if settings.llm_provider == "echo":
        text, used_chunks = answer_from_chunks(question, chunks)
        return RagAnswer(
            question=question,
            answer=text,
            citations=format_citations(used_chunks),
            chunks=used_chunks,
        )

    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=chunks)
    text = invoke_llm(prompt)
    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(chunks),
        chunks=chunks,
    )
