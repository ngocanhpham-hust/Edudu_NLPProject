"""PDF parsing, conservative cleaning, and configurable chunking strategies."""
from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Iterable

import fitz  # PyMuPDF

from .config import settings
from .schemas import Chunk, PageText

_INLINE_SPACE_RE = re.compile(r"[^\S\n]+")
_PARAGRAPH_BREAK_RE = re.compile(r"\n{2,}")
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?。！？])\s+(?=(?:[A-Z0-9À-Ỵ]))", re.UNICODE)


def clean_text(text: str) -> str:
    """Normalize noisy PDF text while preserving line and paragraph structure."""
    text = text.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    text = _INLINE_SPACE_RE.sub(" ", text)
    lines = [line.strip() for line in text.split("\n")]
    cleaned = "\n".join(lines)
    cleaned = _PARAGRAPH_BREAK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def document_id_for_path(path: Path) -> str:
    """Stable id based on file bytes, not path metadata such as mtime."""
    h = hashlib.sha1()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()[:16]


def discover_pdfs(data_dir: Path | str = settings.data_dir) -> list[Path]:
    data_dir = Path(data_dir)
    if not data_dir.exists():
        return []
    return sorted(data_dir.glob("*.pdf"))


def parse_pdf(path: Path | str) -> list[PageText]:
    path = Path(path)
    doc_id = document_id_for_path(path)
    pages: list[PageText] = []
    with fitz.open(path) as pdf:
        for idx, page in enumerate(pdf, start=1):
            text = clean_text(page.get_text("text"))
            if text:
                pages.append(PageText(document_id=doc_id, filename=path.name, page=idx, text=text))
    return pages


def _word_windows(words: list[str], size: int, overlap: int) -> Iterable[tuple[int, int]]:
    if not words:
        return
    start = 0
    step = max(1, size - overlap)
    while start < len(words):
        end = min(len(words), start + size)
        yield start, end
        if end == len(words):
            break
        start += step


def _word_count(text: str) -> int:
    return len(text.split())


def _split_long_text(text: str, max_words: int) -> list[str]:
    words = text.split()
    if len(words) <= max_words:
        return [text.strip()] if text.strip() else []
    return [" ".join(words[start:end]).strip() for start, end in _word_windows(words, max_words, 0)]


def _split_sentences(text: str, max_words: int) -> list[str]:
    pieces = [piece.strip() for piece in _SENTENCE_BOUNDARY_RE.split(text) if piece.strip()]
    units: list[str] = []
    for piece in pieces or [text.strip()]:
        units.extend(_split_long_text(piece, max_words))
    return units


def _paragraph_units(text: str, max_words: int) -> list[str]:
    """Prefer whole paragraphs and split only paragraphs that exceed max_words."""
    units: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        units.extend(_split_long_text(paragraph, max_words))
    return units


def _semantic_units(text: str, max_words: int) -> list[str]:
    """Prefer paragraphs, then lines/sentences, before falling back to word windows."""
    units: list[str] = []
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    for paragraph in paragraphs:
        if _word_count(paragraph) <= max_words:
            units.append(paragraph)
            continue

        lines = [line.strip() for line in paragraph.split("\n") if line.strip()]
        if len(lines) > 1:
            for line in lines:
                units.extend(_split_sentences(line, max_words))
        else:
            units.extend(_split_sentences(paragraph, max_words))
    return units


def _overlap_tail(units: list[str], overlap_words: int) -> list[str]:
    if overlap_words <= 0 or not units:
        return []

    selected: list[str] = []
    selected_words = 0
    for unit in reversed(units):
        words = unit.split()
        word_count = len(words)
        if not selected and word_count > overlap_words:
            selected.append(" ".join(words[-overlap_words:]))
            break
        if selected and selected_words + word_count > overlap_words:
            break
        selected.append(unit)
        selected_words += word_count
    return list(reversed(selected))


def chunk_page(
    page: PageText,
    chunk_size_words: int,
    chunk_overlap_words: int,
    strategy: str = "semantic",
) -> list[Chunk]:
    strategy = strategy.lower().strip()
    if strategy in {"naive", "word", "word_window", "word-window"}:
        words = page.text.split()
        units = [" ".join(words[start:end]) for start, end in _word_windows(words, chunk_size_words, chunk_overlap_words)]
        chunk_overlap_words = 0
    elif strategy in {"paragraph", "paragraph_aware", "paragraph-aware"}:
        units = _paragraph_units(page.text, chunk_size_words)
    elif strategy in {"semantic", "sentence", "sentence_aware", "sentence-aware"}:
        units = _semantic_units(page.text, chunk_size_words)
    else:
        raise ValueError("chunking strategy must be one of: naive, paragraph, semantic")

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0

    def flush() -> None:
        nonlocal current, current_words
        text = "\n\n".join(unit.strip() for unit in current if unit.strip()).strip()
        if text:
            chunk_id = f"{page.document_id}:p{page.page}:c{len(chunks)}"
            chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    document_id=page.document_id,
                    filename=page.filename,
                    page=page.page,
                    text=text,
                    page_start=page.page,
                    page_end=page.page,
                )
            )
        current = _overlap_tail(current, chunk_overlap_words)
        current_words = sum(_word_count(unit) for unit in current)

    for unit in units:
        unit_words = _word_count(unit)
        if current and current_words + unit_words > chunk_size_words:
            flush()
        current.append(unit)
        current_words += unit_words

    if current:
        flush()
    return chunks


def build_chunks_from_pdfs(
    pdf_paths: Iterable[Path | str],
    chunk_size_words: int = settings.chunk_size_words,
    chunk_overlap_words: int = settings.chunk_overlap_words,
    strategy: str = "semantic",
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in pdf_paths:
        for page in parse_pdf(path):
            chunks.extend(chunk_page(page, chunk_size_words, chunk_overlap_words, strategy=strategy))
    return chunks


def build_chunks_from_data_dir(
    data_dir: Path | str = settings.data_dir,
    chunk_size_words: int = settings.chunk_size_words,
    chunk_overlap_words: int = settings.chunk_overlap_words,
    strategy: str = "semantic",
) -> list[Chunk]:
    return build_chunks_from_pdfs(discover_pdfs(data_dir), chunk_size_words, chunk_overlap_words, strategy=strategy)
