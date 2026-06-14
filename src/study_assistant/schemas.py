"""Shared data structures."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PageText:
    document_id: str
    filename: str
    page: int
    text: str


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page: int
    text: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    source_marker: str = ""
    bm25_score: float | None = None
    dense_score: float | None = None
    hybrid_score: float | None = None
    rerank_score: float | None = None

    @property
    def text(self) -> str:
        return self.chunk.text

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["text_preview"] = self.text[:400]
        return data


@dataclass
class Citation:
    source_marker: str
    filename: str
    page: int
    chunk_id: str
    page_start: int | None = None
    page_end: int | None = None


@dataclass
class RagAnswer:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SummaryResult:
    topic: str
    summary: str
    key_points: list[str] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class QuizItem:
    question: str
    options: list[str]
    correct_index: int
    explanation: str
    source_markers: list[str] = field(default_factory=list)


@dataclass
class QuizSet:
    topic: str
    items: list[QuizItem]
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)


@dataclass
class Flashcard:
    front: str
    back: str
    hint: str | None = None
    source_markers: list[str] = field(default_factory=list)


@dataclass
class FlashcardSet:
    topic: str
    cards: list[Flashcard]
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
