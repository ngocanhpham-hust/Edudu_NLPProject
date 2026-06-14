"""Gemini-backed grounded generation for Q&A, extraction, summary, quiz, and flashcards."""
from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .config import settings
from .schemas import Citation, Chunk, Flashcard, FlashcardSet, QuizItem, QuizSet, RagAnswer, RetrievedChunk, SummaryResult

_SOURCE_MARKER_RE = re.compile(r"\[(S\d+)\]")
_MARKDOWN_STRUCTURE_RE = re.compile(r"(^|\n)\s*(?:[-*+]\s+|\d+[.)]\s+|#{1,6}\s+|>\s+)", re.MULTILINE)
_ANSWER_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+(?=(?:[A-Z0-9À-Ỵ]))", re.UNICODE)
_RETRY_DELAY_RE = re.compile(r"retry_delay\s*\{\s*seconds:\s*(\d+)", re.IGNORECASE)


def _assign_source_markers(chunks: list[RetrievedChunk]) -> None:
    for i, item in enumerate(chunks, start=1):
        item.source_marker = item.source_marker or f"S{i}"


def source_markers_in_text(*texts: str) -> list[str]:
    markers: list[str] = []
    seen: set[str] = set()
    for text in texts:
        for marker in _SOURCE_MARKER_RE.findall(text):
            if marker not in seen:
                markers.append(marker)
                seen.add(marker)
    return markers


def validate_source_markers(text: str, valid_markers: set[str]) -> list[str]:
    used = source_markers_in_text(text)
    warnings: list[str] = []
    if not used:
        warnings.append("No source marker like [S1] was found in the generated answer.")
    invalid = [marker for marker in used if marker not in valid_markers]
    if invalid:
        warnings.append(f"Invalid source marker(s): {', '.join(invalid)}.")
    return warnings


def build_citations(chunks: list[RetrievedChunk], used_markers: list[str] | set[str] | None = None) -> list[Citation]:
    _assign_source_markers(chunks)
    used = set(used_markers) if used_markers is not None else None
    citations: list[Citation] = []
    for item in chunks:
        marker = item.source_marker
        if used is not None and marker not in used:
            continue
        page_start = item.chunk.page_start or item.chunk.page
        page_end = item.chunk.page_end or item.chunk.page
        citations.append(
            Citation(
                source_marker=marker,
                filename=item.chunk.filename,
                page=item.chunk.page,
                chunk_id=item.chunk.chunk_id,
                page_start=page_start,
                page_end=page_end,
            )
        )
    return citations


def format_context(chunks: list[RetrievedChunk], max_chars_per_chunk: int = settings.max_chars_per_chunk) -> str:
    _assign_source_markers(chunks)
    lines: list[str] = []
    for item in chunks:
        marker = item.source_marker
        text = item.text[:max_chars_per_chunk].strip()
        lines.append(f"[{marker}] file={item.chunk.filename}, page={item.chunk.page}\n{text}")
    return "\n\n".join(lines)


def retrieved_from_chunks(chunks: list[Chunk]) -> list[RetrievedChunk]:
    return [RetrievedChunk(chunk=chunk, score=0.0, source_marker=f"S{i}") for i, chunk in enumerate(chunks, start=1)]


def _batched(items: list[RetrievedChunk], batch_size: int) -> list[list[RetrievedChunk]]:
    return [items[i : i + batch_size] for i in range(0, len(items), max(1, batch_size))]


def _retry_delay_seconds(exc: Exception, attempt: int) -> float:
    """Honor Gemini quota retry hints when they are present in API errors."""
    text = str(exc)
    match = _RETRY_DELAY_RE.search(text)
    if match:
        return float(match.group(1)) + 1.0
    return 1.5 * (attempt + 1)


class GeminiClient:
    """Small Gemini wrapper with disk cache for reproducible demos."""

    def __init__(self, api_key: str = settings.google_api_key, model_name: str = settings.gemini_model):
        self.api_key = api_key
        self.model_name = model_name
        self.cache_path = settings.gemini_cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache: dict[str, str] = {}
        if self.cache_path.exists():
            try:
                self.cache = json.loads(self.cache_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                self.cache = {}
        self._model = None

    def _load_model(self):
        if not self.api_key:
            raise RuntimeError("GOOGLE_API_KEY is missing. Set it in .env, shell, or Kaggle Secrets.")
        if self._model is None:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        return self._model

    def generate(self, prompt: str, use_cache: bool = True) -> str:
        key = hashlib.sha1((self.model_name + "\n" + prompt).encode("utf-8")).hexdigest()
        if use_cache and key in self.cache:
            return self.cache[key]
        last_error: Exception | None = None
        for attempt in range(settings.max_retries):
            try:
                response = self._load_model().generate_content(prompt)
                text = getattr(response, "text", None) or str(response)
                self.cache[key] = text
                self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")
                return text
            except Exception as exc:  # network/quota/transient API failures
                last_error = exc
                if attempt < settings.max_retries - 1:
                    time.sleep(_retry_delay_seconds(exc, attempt))
        raise RuntimeError(f"Gemini call failed after {settings.max_retries} retries: {last_error}")


def _json_from_text(text: str) -> Any:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return json.loads(cleaned[start : end + 1])
        raise


def format_answer_for_readability(answer: str) -> str:
    """Keep model Markdown, otherwise split long plain answers into short paragraphs."""
    answer = answer.strip()
    if not answer:
        return answer
    if "\n\n" in answer or _MARKDOWN_STRUCTURE_RE.search(answer):
        return answer

    sentences = [sentence.strip() for sentence in _ANSWER_SENTENCE_BOUNDARY_RE.split(answer) if sentence.strip()]
    if len(sentences) < 4 and len(answer) < 700:
        return answer
    if len(sentences) < 2:
        return answer

    paragraph_size = 2
    paragraphs = [" ".join(sentences[i : i + paragraph_size]) for i in range(0, len(sentences), paragraph_size)]
    return "\n\n".join(paragraphs)


def answer_question(question: str, chunks: list[RetrievedChunk], client: GeminiClient | None = None) -> RagAnswer:
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Không tìm thấy ngữ cảnh phù hợp trong tài liệu.",
            warnings=["No retrieved chunks were available."],
        )
    client = client or GeminiClient()
    context = format_context(chunks)
    prompt = f"""
You are a document-grounded study assistant.
Answer the question using ONLY the provided context.
If the answer is not in the context, say that the document does not provide enough information.
Cite sources using markers like [S1], [S2].
Return readable Markdown:
- Use bullet points when the answer contains multiple facts, reasons, steps, definitions, or examples.
- Otherwise use short paragraphs. Do not return one long paragraph.
- Keep each bullet or paragraph concise and cite key claims with [S1], [S2], etc.
- Answer in the same language as the question.

Question:
{question}

Context:
{context}

Answer:
""".strip()
    answer = format_answer_for_readability(client.generate(prompt))
    valid_markers = {item.source_marker for item in chunks}
    used_markers = source_markers_in_text(answer)
    warnings = validate_source_markers(answer, valid_markers)
    return RagAnswer(question=question, answer=answer, citations=build_citations(chunks, used_markers), chunks=chunks, warnings=warnings)


def extract_information(request: str, chunks: list[RetrievedChunk], client: GeminiClient | None = None) -> RagAnswer:
    if not chunks:
        return RagAnswer(
            question=request,
            answer="Không tìm thấy ngữ cảnh phù hợp trong tài liệu.",
            warnings=["No retrieved chunks were available."],
        )
    client = client or GeminiClient()
    context = format_context(chunks)
    prompt = f"""
Extract information requested by the user from ONLY the context.
Return concise bullet points. Cite each important point using [S1], [S2], etc.

Request:
{request}

Context:
{context}
""".strip()
    answer = client.generate(prompt).strip()
    valid_markers = {item.source_marker for item in chunks}
    used_markers = source_markers_in_text(answer)
    warnings = validate_source_markers(answer, valid_markers)
    return RagAnswer(question=request, answer=answer, citations=build_citations(chunks, used_markers), chunks=chunks, warnings=warnings)


def summarize_topic(topic: str, chunks: list[RetrievedChunk], client: GeminiClient | None = None) -> SummaryResult:
    if not chunks:
        return SummaryResult(topic=topic, summary="", key_points=[], citations=[], chunks=[])
    client = client or GeminiClient()
    context = format_context(chunks)
    prompt = f"""
Summarize the topic using ONLY the provided context.
Return valid JSON only, with this schema:
{{
  "summary": "short grounded summary",
  "key_points": ["point 1 [S1]", "point 2 [S2]"]
}}

Topic:
{topic}

Context:
{context}
""".strip()
    payload = _json_from_text(client.generate(prompt))
    summary = str(payload.get("summary", "")).strip()
    key_points = [str(x).strip() for x in payload.get("key_points", []) if str(x).strip()]
    used_markers = source_markers_in_text(summary, *key_points)
    return SummaryResult(topic=topic, summary=summary, key_points=key_points, citations=build_citations(chunks, used_markers), chunks=chunks)


def summarize_document(
    topic: str,
    chunks: list[RetrievedChunk],
    client: GeminiClient | None = None,
    batch_size: int = settings.summary_batch_size,
) -> SummaryResult:
    """Map-reduce style full-document summarization over all provided chunks."""
    if not chunks:
        return SummaryResult(topic=topic, summary="", key_points=[], citations=[], chunks=[])

    client = client or GeminiClient()
    _assign_source_markers(chunks)
    partials: list[str] = []
    for batch_id, batch in enumerate(_batched(chunks, batch_size), start=1):
        context = format_context(batch)
        prompt = f"""
Summarize this document part using ONLY the provided context.
Return valid JSON only, with this schema:
{{
  "summary": "short grounded partial summary",
  "key_points": ["point 1 [S1]", "point 2 [S2]"]
}}

Overall topic:
{topic}

Context:
{context}
""".strip()
        payload = _json_from_text(client.generate(prompt))
        summary = str(payload.get("summary", "")).strip()
        key_points = [str(x).strip() for x in payload.get("key_points", []) if str(x).strip()]
        partials.append(
            "\n".join(
                [
                    f"Part {batch_id}: {summary}",
                    *[f"- {point}" for point in key_points],
                ]
            )
        )

    reduce_prompt = f"""
Combine the grounded partial summaries into one full-document summary.
Use only the provided partial summaries. Preserve source markers like [S1], [S2] when making claims.
Return valid JSON only, with this schema:
{{
  "summary": "full document summary with source markers",
  "key_points": ["important point [S1]", "important point [S2]"]
}}

Topic:
{topic}

Partial summaries:
{chr(10).join(partials)}
""".strip()
    payload = _json_from_text(client.generate(reduce_prompt))
    summary = str(payload.get("summary", "")).strip()
    key_points = [str(x).strip() for x in payload.get("key_points", []) if str(x).strip()]
    used_markers = source_markers_in_text(summary, *key_points)
    return SummaryResult(topic=topic, summary=summary, key_points=key_points, citations=build_citations(chunks, used_markers), chunks=chunks)


def validate_quiz_payload(payload: Any, valid_markers: set[str]) -> list[QuizItem]:
    raw_items = payload.get("items", []) if isinstance(payload, dict) else []
    items: list[QuizItem] = []
    seen: set[str] = set()
    for raw in raw_items:
        try:
            question = str(raw["question"]).strip()
            options = [str(x).strip() for x in raw["options"]]
            correct_index = int(raw["correct_index"])
            explanation = str(raw.get("explanation", "")).strip()
            markers = [str(m).strip() for m in raw.get("source_markers", []) if str(m).strip() in valid_markers]
        except Exception:
            continue
        norm = question.lower()
        if not question or norm in seen or len(options) != 4 or not 0 <= correct_index < 4 or not explanation or not markers:
            continue
        seen.add(norm)
        items.append(QuizItem(question=question, options=options, correct_index=correct_index, explanation=explanation, source_markers=markers))
    return items


def generate_quiz(topic: str, chunks: list[RetrievedChunk], count: int = settings.quiz_default_count, client: GeminiClient | None = None) -> QuizSet:
    if not chunks:
        return QuizSet(topic=topic, items=[], citations=[], chunks=[])
    client = client or GeminiClient()
    context = format_context(chunks)
    prompt = f"""
Create {count} multiple-choice questions from ONLY the context.
Return valid JSON only:
{{
  "items": [
    {{
      "question": "...",
      "options": ["A", "B", "C", "D"],
      "correct_index": 0,
      "explanation": "why the answer is correct, with source marker",
      "source_markers": ["S1"]
    }}
  ]
}}
Every question must be answerable from the context.

Topic:
{topic}

Context:
{context}
""".strip()
    payload = _json_from_text(client.generate(prompt))
    valid_markers = {item.source_marker for item in chunks}
    items = validate_quiz_payload(payload, valid_markers)
    used_markers = {marker for item in items for marker in item.source_markers}
    return QuizSet(topic=topic, items=items, citations=build_citations(chunks, used_markers), chunks=chunks)


def validate_flashcard_payload(payload: Any, valid_markers: set[str]) -> list[Flashcard]:
    raw_cards = payload.get("cards", []) if isinstance(payload, dict) else []
    cards: list[Flashcard] = []
    seen: set[str] = set()
    for raw in raw_cards:
        front = str(raw.get("front", "")).strip()
        back = str(raw.get("back", "")).strip()
        hint = str(raw.get("hint", "")).strip() or None
        markers = [str(m).strip() for m in raw.get("source_markers", []) if str(m).strip() in valid_markers]
        norm = front.lower()
        if not front or not back or norm in seen or not markers:
            continue
        seen.add(norm)
        cards.append(Flashcard(front=front, back=back, hint=hint, source_markers=markers))
    return cards


def generate_flashcards(topic: str, chunks: list[RetrievedChunk], count: int = settings.flashcard_default_count, client: GeminiClient | None = None) -> FlashcardSet:
    if not chunks:
        return FlashcardSet(topic=topic, cards=[], citations=[], chunks=[])
    client = client or GeminiClient()
    context = format_context(chunks)
    prompt = f"""
Create {count} flashcards from ONLY the context.
Return valid JSON only:
{{
  "cards": [
    {{"front": "...", "back": "...", "hint": "...", "source_markers": ["S1"]}}
  ]
}}

Topic:
{topic}

Context:
{context}
""".strip()
    payload = _json_from_text(client.generate(prompt))
    valid_markers = {item.source_marker for item in chunks}
    cards = validate_flashcard_payload(payload, valid_markers)
    used_markers = {marker for card in cards for marker in card.source_markers}
    return FlashcardSet(topic=topic, cards=cards, citations=build_citations(chunks, used_markers), chunks=chunks)


def result_to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    return obj
