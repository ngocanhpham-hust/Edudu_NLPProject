"""Central configuration for local demo and Kaggle experiments."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _str(name: str, default: str) -> str:
    return os.getenv(name, os.getenv(f"STUDY_{name}", default))


def _int(name: str, default: int) -> int:
    value = os.getenv(name, os.getenv(f"STUDY_{name}", ""))
    return default if value == "" else int(value)


def _float(name: str, default: float) -> float:
    value = os.getenv(name, os.getenv(f"STUDY_{name}", ""))
    return default if value == "" else float(value)


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name, os.getenv(f"STUDY_{name}", ""))
    if value == "":
        return default
    return value.lower() in {"1", "true", "yes", "y"}


@dataclass(frozen=True)
class Settings:
    # Paths
    data_dir: Path = Path(_str("DATA_DIR", "data"))
    outputs_dir: Path = Path(_str("OUTPUTS_DIR", "outputs"))
    index_dir: Path = Path(_str("INDEX_DIR", "indexes/faiss_index"))

    # PDF/chunking
    chunk_size_words: int = _int("CHUNK_SIZE_WORDS", 700)
    chunk_overlap_words: int = _int("CHUNK_OVERLAP_WORDS", 100)

    # Retrieval
    embedding_model: str = _str(
        "EMBEDDING_MODEL",
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    stronger_embedding_model: str = _str("STRONGER_EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
    bm25_tokenizer: str = _str("BM25_TOKENIZER", "regex")
    hybrid_alpha: float = _float("HYBRID_ALPHA", 0.70)
    rrf_k: int = _int("RRF_K", 60)
    dense_top_k: int = _int("DENSE_TOP_K", 15)
    final_top_k: int = _int("FINAL_TOP_K", 5)
    candidate_k: int = _int("CANDIDATE_K", 30)

    # Optional reranker
    use_reranker: bool = _bool("USE_RERANKER", False)
    reranker_model: str = _str("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_initial_k: int = _int("RERANK_INITIAL_K", 30)
    rerank_k: int = _int("RERANK_K", 5)

    # Gemini generation
    google_api_key: str = _str("GOOGLE_API_KEY", "")
    gemini_model: str = _str("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_cache_path: Path = Path(_str("GEMINI_CACHE_PATH", "outputs/gemini_cache.json"))
    max_retries: int = _int("MAX_RETRIES", 3)
    max_chars_per_chunk: int = _int("MAX_CHARS_PER_CHUNK", 1200)

    # Generation task defaults
    summary_retrieval_k: int = _int("SUMMARY_RETRIEVAL_K", 8)
    summary_batch_size: int = _int("SUMMARY_BATCH_SIZE", 8)
    quiz_default_count: int = _int("QUIZ_DEFAULT_COUNT", 5)
    flashcard_default_count: int = _int("FLASHCARD_DEFAULT_COUNT", 8)

    def validate(self) -> None:
        if self.chunk_size_words <= 0:
            raise ValueError("chunk_size_words must be positive")
        if self.chunk_overlap_words < 0:
            raise ValueError("chunk_overlap_words must be non-negative")
        if self.chunk_overlap_words >= self.chunk_size_words:
            raise ValueError("chunk_overlap_words must be smaller than chunk_size_words")
        if not 0.0 <= self.hybrid_alpha <= 1.0:
            raise ValueError("hybrid_alpha must be in [0, 1]")


settings = Settings()
settings.validate()
