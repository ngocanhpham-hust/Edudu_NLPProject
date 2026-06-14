"""BM25, dense FAISS retrieval, hybrid fusion, and cross-encoder reranking."""
from __future__ import annotations

import json
import pickle
import re
from functools import lru_cache
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from .config import settings
from .document import build_chunks_from_data_dir, build_chunks_from_pdfs
from .schemas import Chunk, RetrievedChunk

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@lru_cache(maxsize=1)
def _vietnamese_segmenter():
    try:
        from underthesea import word_tokenize

        return lambda text: word_tokenize(text, format="text")
    except Exception:
        pass
    try:
        from pyvi import ViTokenizer

        return ViTokenizer.tokenize
    except Exception:
        return None


def tokenize(text: str, mode: str = settings.bm25_tokenizer) -> list[str]:
    mode = mode.lower().strip()
    if mode in {"vi", "vietnamese", "vietnamese_word_segmentation"}:
        segmenter = _vietnamese_segmenter()
        if segmenter is not None:
            text = segmenter(text.lower())
    elif mode not in {"regex", "simple"}:
        raise ValueError("BM25 tokenizer must be one of: regex, simple, vi, vietnamese")
    return _TOKEN_RE.findall(text.lower())


def minmax(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    values = values.astype("float32")
    if values.size == 0:
        return values
    vmin = float(values.min())
    vmax = float(values.max())
    if abs(vmax - vmin) < eps:
        return np.zeros_like(values, dtype="float32")
    return (values - vmin) / (vmax - vmin + eps)


def keyword_overlap_ratio(query: str, text: str) -> float:
    q = set(tokenize(query))
    if not q:
        return 0.0
    c = set(tokenize(text))
    return len(q & c) / max(1, len(q))


class StudyIndex:
    """In-memory + serializable retrieval index.

    Live default: hybrid retrieval = alpha * normalized dense + (1-alpha) * normalized BM25.
    RRF is also available as a scale-free fusion method.
    """

    def __init__(
        self,
        embedding_model_name: str = settings.embedding_model,
        bm25_tokenizer: str = settings.bm25_tokenizer,
    ):
        self.embedding_model_name = embedding_model_name
        self.bm25_tokenizer = bm25_tokenizer
        self.embedding_model: SentenceTransformer | None = None
        self.chunks: list[Chunk] = []
        self.bm25: BM25Okapi | None = None
        self.index: faiss.IndexFlatIP | None = None
        self.embeddings: np.ndarray | None = None
        self._reranker = None

    def _model(self) -> SentenceTransformer:
        if self.embedding_model is None:
            self.embedding_model = SentenceTransformer(self.embedding_model_name)
        return self.embedding_model

    def _uses_e5_prefixes(self) -> bool:
        return "e5" in self.embedding_model_name.lower()

    def _encode_passages(self, texts: list[str]) -> np.ndarray:
        if self._uses_e5_prefixes():
            texts = [text if text.lower().startswith("passage:") else f"passage: {text}" for text in texts]
        vectors = self._model().encode(texts, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
        return np.asarray(vectors, dtype="float32")

    def _encode_query(self, query: str) -> np.ndarray:
        if self._uses_e5_prefixes() and not query.lower().startswith("query:"):
            query = f"query: {query}"
        vector = self._model().encode([query], normalize_embeddings=True)
        return np.asarray(vector, dtype="float32")

    def build(self, chunks: list[Chunk]) -> "StudyIndex":
        if not chunks:
            raise ValueError("No chunks found. Put PDF files into data/ or pass pdf_paths.")
        self.chunks = chunks
        tokenized_corpus = [tokenize(c.text, self.bm25_tokenizer) for c in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        texts = [c.text for c in chunks]
        self.embeddings = self._encode_passages(texts)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)
        return self

    def build_from_data_dir(
        self,
        data_dir: Path | str = settings.data_dir,
        chunk_size_words: int = settings.chunk_size_words,
        chunk_overlap_words: int = settings.chunk_overlap_words,
        chunking_strategy: str = "semantic",
    ) -> "StudyIndex":
        chunks = build_chunks_from_data_dir(data_dir, chunk_size_words, chunk_overlap_words, strategy=chunking_strategy)
        return self.build(chunks)

    def build_from_pdfs(
        self,
        pdf_paths: list[Path | str],
        chunk_size_words: int = settings.chunk_size_words,
        chunk_overlap_words: int = settings.chunk_overlap_words,
        chunking_strategy: str = "semantic",
    ) -> "StudyIndex":
        chunks = build_chunks_from_pdfs(pdf_paths, chunk_size_words, chunk_overlap_words, strategy=chunking_strategy)
        return self.build(chunks)

    def save(self, index_dir: Path | str = settings.index_dir) -> None:
        if self.index is None or self.embeddings is None or self.bm25 is None:
            raise RuntimeError("Index has not been built.")
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "vectors.faiss"))
        (index_dir / "chunks.jsonl").write_text(
            "\n".join(json.dumps(c.__dict__, ensure_ascii=False) for c in self.chunks),
            encoding="utf-8",
        )
        with (index_dir / "bm25.pkl").open("wb") as f:
            pickle.dump(self.bm25, f)
        (index_dir / "meta.json").write_text(
            json.dumps(
                {
                    "embedding_model_name": self.embedding_model_name,
                    "bm25_tokenizer": self.bm25_tokenizer,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, index_dir: Path | str = settings.index_dir) -> "StudyIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        obj = cls(
            embedding_model_name=meta["embedding_model_name"],
            bm25_tokenizer=meta.get("bm25_tokenizer", settings.bm25_tokenizer),
        )
        obj.index = faiss.read_index(str(index_dir / "vectors.faiss"))
        obj.chunks = [Chunk(**json.loads(line)) for line in (index_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        with (index_dir / "bm25.pkl").open("rb") as f:
            obj.bm25 = pickle.load(f)
        return obj

    def _check_ready(self) -> None:
        if self.bm25 is None or self.index is None or not self.chunks:
            raise RuntimeError("Index is not ready. Build or load it first.")

    def bm25_scores(self, query: str) -> np.ndarray:
        self._check_ready()
        assert self.bm25 is not None
        return np.asarray(self.bm25.get_scores(tokenize(query, self.bm25_tokenizer)), dtype="float32")

    def dense_scores(self, query: str, candidate_k: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        self._check_ready()
        assert self.index is not None
        q = self._encode_query(query)
        k = min(candidate_k or len(self.chunks), len(self.chunks))
        scores, ids = self.index.search(q, k)
        return ids[0], scores[0].astype("float32")

    def retrieve_bm25(self, query: str, k: int = settings.final_top_k) -> list[RetrievedChunk]:
        scores = self.bm25_scores(query)
        top_ids = np.argsort(scores)[::-1][:k]
        return self._make_results(top_ids, scores[top_ids], "bm25")

    def retrieve_dense(self, query: str, k: int = settings.final_top_k) -> list[RetrievedChunk]:
        ids, scores = self.dense_scores(query, k)
        return self._make_results(ids, scores, "dense")

    def retrieve_hybrid(
        self,
        query: str,
        k: int = settings.final_top_k,
        alpha: float = settings.hybrid_alpha,
        candidate_k: int = settings.candidate_k,
    ) -> list[RetrievedChunk]:
        bm25_all = self.bm25_scores(query)
        dense_ids, dense_scores = self.dense_scores(query, min(candidate_k, len(self.chunks)))
        bm25_ids = np.argsort(bm25_all)[::-1][: min(candidate_k, len(self.chunks))]
        candidate_ids = np.array(sorted(set(map(int, dense_ids)) | set(map(int, bm25_ids))), dtype="int64")

        dense_map = {int(i): float(s) for i, s in zip(dense_ids, dense_scores)}
        dense_raw = np.array([dense_map.get(int(i), 0.0) for i in candidate_ids], dtype="float32")
        bm25_raw = bm25_all[candidate_ids]

        dense_norm = minmax(dense_raw)
        bm25_norm = minmax(bm25_raw)
        hybrid = alpha * dense_norm + (1.0 - alpha) * bm25_norm
        order = np.argsort(hybrid)[::-1][:k]

        results: list[RetrievedChunk] = []
        for rank, pos in enumerate(order, start=1):
            idx = int(candidate_ids[pos])
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(hybrid[pos]),
                    source_marker=f"S{rank}",
                    bm25_score=float(bm25_norm[pos]),
                    dense_score=float(dense_norm[pos]),
                    hybrid_score=float(hybrid[pos]),
                )
            )
        return results

    def retrieve_rrf(
        self,
        query: str,
        k: int = settings.final_top_k,
        candidate_k: int = settings.candidate_k,
        rrf_k: int = settings.rrf_k,
    ) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion over BM25 and dense ranks.

        RRF is scale-free, so it avoids the per-query score calibration issue of min-max fusion.
        """
        bm25_all = self.bm25_scores(query)
        limit = min(candidate_k, len(self.chunks))
        bm25_ids = np.argsort(bm25_all)[::-1][:limit]
        dense_ids, dense_scores = self.dense_scores(query, limit)

        bm25_rank = {int(idx): rank for rank, idx in enumerate(bm25_ids, start=1)}
        dense_rank = {int(idx): rank for rank, idx in enumerate(dense_ids, start=1)}
        dense_score_map = {int(idx): float(score) for idx, score in zip(dense_ids, dense_scores)}
        candidate_ids = sorted(set(bm25_rank) | set(dense_rank))

        fused: list[tuple[int, float]] = []
        for idx in candidate_ids:
            score = 0.0
            if idx in bm25_rank:
                score += 1.0 / (rrf_k + bm25_rank[idx])
            if idx in dense_rank:
                score += 1.0 / (rrf_k + dense_rank[idx])
            fused.append((idx, score))
        fused.sort(key=lambda item: item[1], reverse=True)

        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(fused[:k], start=1):
            results.append(
                RetrievedChunk(
                    chunk=self.chunks[idx],
                    score=float(score),
                    source_marker=f"S{rank}",
                    bm25_score=float(bm25_all[idx]),
                    dense_score=dense_score_map.get(idx),
                    hybrid_score=float(score),
                )
            )
        return results

    def retrieve(
        self,
        query: str,
        method: str = "hybrid",
        k: int = settings.final_top_k,
        alpha: float = settings.hybrid_alpha,
        rerank: bool = False,
        candidate_k: int = settings.candidate_k,
        rerank_initial_k: int | None = None,
    ) -> list[RetrievedChunk]:
        if method == "bm25":
            results = self.retrieve_bm25(query, k)
        elif method == "dense":
            results = self.retrieve_dense(query, k)
        elif method == "hybrid":
            initial_k = rerank_initial_k or settings.rerank_initial_k if rerank else k
            results = self.retrieve_hybrid(query, initial_k, alpha=alpha, candidate_k=candidate_k)
        elif method == "rrf":
            initial_k = rerank_initial_k or settings.rerank_initial_k if rerank else k
            results = self.retrieve_rrf(query, initial_k, candidate_k=candidate_k)
        else:
            raise ValueError("method must be one of: bm25, dense, hybrid, rrf")
        if rerank:
            return self.rerank(query, results, rerank_k=k)
        return results

    def retrieve_final(
        self,
        query: str,
        k: int = settings.final_top_k,
        alpha: float = settings.hybrid_alpha,
        candidate_k: int = settings.candidate_k,
        rerank_initial_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Final pipeline retriever: hybrid min-max candidates followed by cross-encoder reranking."""
        initial_k = rerank_initial_k or candidate_k
        initial = self.retrieve_hybrid(query, k=initial_k, alpha=alpha, candidate_k=candidate_k)
        return self.rerank(query, initial, rerank_k=k)

    def rerank(self, query: str, chunks: list[RetrievedChunk], rerank_k: int = settings.rerank_k) -> list[RetrievedChunk]:
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            self._reranker = CrossEncoder(settings.reranker_model)
        pairs = [[query, r.text] for r in chunks]
        scores = self._reranker.predict(pairs)
        for r, s in zip(chunks, scores):
            r.rerank_score = float(s)
            r.score = float(s)
        ranked = sorted(chunks, key=lambda r: r.score, reverse=True)[:rerank_k]
        for i, r in enumerate(ranked, start=1):
            r.source_marker = f"S{i}"
        return ranked

    def _make_results(self, ids: np.ndarray, scores: np.ndarray, score_type: str) -> list[RetrievedChunk]:
        results: list[RetrievedChunk] = []
        for rank, (idx, score) in enumerate(zip(ids, scores), start=1):
            idx = int(idx)
            r = RetrievedChunk(chunk=self.chunks[idx], score=float(score), source_marker=f"S{rank}")
            if score_type == "bm25":
                r.bm25_score = float(score)
            elif score_type == "dense":
                r.dense_score = float(score)
            results.append(r)
        return results

    def features_for_query_chunk(self, query: str, chunk_idx: int) -> list[float]:
        bm25_all = self.bm25_scores(query)
        dense_ids, dense_raw = self.dense_scores(query, len(self.chunks))
        dense_map = {int(i): float(s) for i, s in zip(dense_ids, dense_raw)}
        bm25_norm = minmax(bm25_all)[chunk_idx]
        dense_norm = dense_map.get(chunk_idx, 0.0)
        length_norm = min(1.0, len(self.chunks[chunk_idx].text.split()) / max(1, settings.chunk_size_words))
        overlap = keyword_overlap_ratio(query, self.chunks[chunk_idx].text)
        return [float(bm25_norm), float(dense_norm), float(length_norm), float(overlap)]
