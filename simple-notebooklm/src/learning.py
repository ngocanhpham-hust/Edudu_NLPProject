import json
import re
from typing import Any
from json import JSONDecodeError

from pydantic import BaseModel, ValidationError

from src.config import settings
from src.llm import invoke_llm
from src.offline import make_flashcards, make_quiz_items, summarize_chunks
from src.rag import fetch_all_chunks, format_citations, render_prompt, retrieve
from src.schemas import Flashcard, FlashcardSet, QuizItem, QuizSet, RetrievedChunk, Summary

SUMMARY_SINGLE_TEMPLATE = "summary_single.jinja2"
SUMMARY_MAP_TEMPLATE = "summary_map.jinja2"
SUMMARY_REDUCE_TEMPLATE = "summary_reduce.jinja2"
QUIZ_TEMPLATE = "quiz.jinja2"
FLASHCARDS_TEMPLATE = "flashcards.jinja2"


def _resolve_target(
    document: str | None,
    query: str | None,
    filters: dict[str, Any] | None,
    k: int | None,
    retrieval_k: int,
) -> tuple[list[RetrievedChunk], str, str | None]:
    effective_filters = dict(filters or {})
    if document:
        effective_filters["filename"] = document
    if query:
        chunks = retrieve(query, k=k or retrieval_k, filters=effective_filters)
        return chunks, "query", query
    if effective_filters:
        chunks = fetch_all_chunks(filters=effective_filters)
        scope = "document" if document else "filter"
        target = ", ".join(f"{key}={value}" for key, value in effective_filters.items())
        return chunks, scope, target
    return fetch_all_chunks(filters=None), "corpus", None


def _extract_json(text: str) -> str:
    cleaned = text.strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        return fence.group(1).strip()
    starts = [idx for idx in (cleaned.find("{"), cleaned.find("[")) if idx >= 0]
    if starts:
        start = min(starts)
        end = max(cleaned.rfind("}"), cleaned.rfind("]"))
        if end > start:
            return cleaned[start : end + 1]
    return cleaned


def _parse_json(text: str) -> dict[str, Any] | list[Any]:
    cleaned = _extract_json(text)
    try:
        obj = json.loads(cleaned)
    except JSONDecodeError:
        decoder = json.JSONDecoder()
        for start, char in enumerate(cleaned):
            if char not in "[{":
                continue
            try:
                obj, _ = decoder.raw_decode(cleaned[start:])
                break
            except JSONDecodeError:
                continue
        else:
            raise
    if not isinstance(obj, dict | list):
        raise RuntimeError("Expected JSON object or array.")
    return obj


def _validate_summary_payload(payload: dict[str, Any] | list[Any]) -> tuple[str, list[str]]:
    if not isinstance(payload, dict):
        raise RuntimeError("Summary payload must be a JSON object.")
    summary = str(payload.get("summary") or "").strip()
    key_points = [str(item).strip() for item in payload.get("key_points", []) if str(item).strip()]
    if not summary:
        raise RuntimeError("No summary produced.")
    return summary, key_points


def _validate_items(
    payload: dict[str, Any] | list[Any],
    key: str,
    model_class: type[BaseModel],
    dedup_field: str,
    label: str,
    valid_markers: set[str],
) -> list[BaseModel]:
    raw_items = payload.get(key) if isinstance(payload, dict) else payload
    if not isinstance(raw_items, list):
        raise RuntimeError(f"Expected a list of {label}.")
    items: list[BaseModel] = []
    seen: set[str] = set()
    for raw in raw_items:
        try:
            item = model_class.model_validate(raw)
        except ValidationError:
            continue
        norm = str(getattr(item, dedup_field, "")).strip().lower()
        if not norm or norm in seen:
            continue
        seen.add(norm)
        markers = [m for m in getattr(item, "source_markers", []) if m in valid_markers]
        items.append(item.model_copy(update={"source_markers": markers}))
    if not items:
        raise RuntimeError(f"No valid {label} produced.")
    return items


def summarize(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, Any] | None = None,
    k: int | None = None,
) -> Summary:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.summarize_retrieval_k
    )
    if not chunks:
        return Summary(scope=scope, target=target, summary="Không tìm thấy ngữ cảnh phù hợp.")

    if settings.llm_provider == "echo":
        summary_text, key_points, used_chunks = summarize_chunks(chunks)
        chunks_for_output = used_chunks
    else:
        if len(chunks) <= settings.summarize_batch_size:
            payload = _parse_json(invoke_llm(render_prompt(SUMMARY_SINGLE_TEMPLATE, chunks=chunks)))
            summary_text, key_points = _validate_summary_payload(payload)
        else:
            partials = []
            for start in range(0, len(chunks), settings.summarize_batch_size):
                batch = chunks[start : start + settings.summarize_batch_size]
                payload = _parse_json(invoke_llm(render_prompt(SUMMARY_MAP_TEMPLATE, chunks=batch)))
                summary_text, key_points = _validate_summary_payload(payload)
                partials.append({"summary": summary_text, "key_points": key_points})
            payload = _parse_json(
                invoke_llm(render_prompt(SUMMARY_REDUCE_TEMPLATE, partials=partials))
            )
            summary_text, key_points = _validate_summary_payload(payload)
        chunks_for_output = chunks

    return Summary(
        scope=scope,
        target=target,
        summary=summary_text,
        key_points=key_points,
        citations=format_citations(chunks_for_output),
        chunks=chunks_for_output,
    )


def generate_quiz(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, Any] | None = None,
    count: int | None = None,
    k: int | None = None,
) -> QuizSet:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    if not chunks:
        return QuizSet(scope=scope, target=target)
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    n = count or settings.quiz_default_count
    if settings.llm_provider == "echo":
        items, chunks_for_output = make_quiz_items(chunks, n)
    else:
        try:
            payload = _parse_json(
                invoke_llm(render_prompt(QUIZ_TEMPLATE, chunks=chunks, count=n))
            )
            items = _validate_items(payload, "items", QuizItem, "question", "quiz items", valid_markers)
            chunks_for_output = chunks
        except (JSONDecodeError, RuntimeError, ValidationError):
            items, chunks_for_output = make_quiz_items(chunks, n)
    return QuizSet(
        scope=scope,
        target=target,
        items=items,
        citations=format_citations(chunks_for_output),
        chunks=chunks_for_output,
    )


def generate_flashcards(
    document: str | None = None,
    query: str | None = None,
    filters: dict[str, Any] | None = None,
    count: int | None = None,
    k: int | None = None,
) -> FlashcardSet:
    chunks, scope, target = _resolve_target(
        document, query, filters, k, settings.generation_retrieval_k
    )
    if not chunks:
        return FlashcardSet(scope=scope, target=target)
    valid_markers = {f"S{i}" for i in range(1, len(chunks) + 1)}
    n = count or settings.flashcards_default_count
    if settings.llm_provider == "echo":
        cards, chunks_for_output = make_flashcards(chunks, n)
    else:
        try:
            payload = _parse_json(
                invoke_llm(
                    render_prompt(
                        FLASHCARDS_TEMPLATE,
                        chunks=chunks,
                        count=n,
                    )
                )
            )
            cards = _validate_items(payload, "cards", Flashcard, "front", "flashcards", valid_markers)
            chunks_for_output = chunks
        except (JSONDecodeError, RuntimeError, ValidationError):
            cards, chunks_for_output = make_flashcards(chunks, n)
    return FlashcardSet(
        scope=scope,
        target=target,
        cards=cards,
        citations=format_citations(chunks_for_output),
        chunks=chunks_for_output,
    )
