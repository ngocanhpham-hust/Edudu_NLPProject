"""Evaluation utilities for Kaggle notebooks: retrieval, generation, and ablations."""
from __future__ import annotations

import ast
import csv
import json
import re
import time
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .document import build_chunks_from_data_dir
from .config import settings
from .retrieval import StudyIndex

_SOURCE_MARKER_RE = re.compile(r"\[(S\d+)\]")


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    p = normalize_answer(prediction).split()
    r = normalize_answer(reference).split()
    if not p or not r:
        return float(p == r)
    common = set(p) & set(r)
    if not common:
        return 0.0
    precision = sum(min(p.count(t), r.count(t)) for t in common) / len(p)
    recall = sum(min(p.count(t), r.count(t)) for t in common) / len(r)
    return 2 * precision * recall / max(1e-9, precision + recall)


def _parse_list_cell(value: str | float | int | None) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    try:
        obj = ast.literal_eval(text)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
    except Exception:
        pass
    return [x.strip() for x in re.split(r"[;,]", text) if x.strip()]


def load_benchmark_csv(path: Path | str) -> list[dict]:
    """Load benchmark with columns: question, answer, relevant_pages, relevant_chunk_ids, question_type."""
    rows: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            question_type = row.get("question_type", "").strip().lower() or "unknown"
            rows.append(
                {
                    "question": row.get("question", "").strip(),
                    "answer": row.get("answer", "").strip(),
                    "relevant_pages": _parse_list_cell(row.get("relevant_pages")),
                    "relevant_chunk_ids": _parse_list_cell(row.get("relevant_chunk_ids")),
                    "question_type": question_type,
                    "answerable": question_type != "unanswerable",
                }
            )
    return [r for r in rows if r["question"]]


def _is_relevant(retrieved, case: dict) -> bool:
    relevant_chunks = set(case.get("relevant_chunk_ids", []))
    relevant_pages = set(case.get("relevant_pages", []))
    chunk = retrieved.chunk
    if chunk.chunk_id in relevant_chunks:
        return True
    page_key_1 = f"{chunk.filename}:{chunk.page}"
    page_key_2 = str(chunk.page)
    return page_key_1 in relevant_pages or page_key_2 in relevant_pages


def _relevant_keys(case: dict) -> set[str]:
    keys = {f"chunk:{chunk_id}" for chunk_id in case.get("relevant_chunk_ids", [])}
    keys.update(f"page:{page}" for page in case.get("relevant_pages", []))
    return keys


def _matched_relevant_keys(retrieved, case: dict) -> set[str]:
    relevant_chunks = set(case.get("relevant_chunk_ids", []))
    relevant_pages = set(case.get("relevant_pages", []))
    chunk = retrieved.chunk
    keys: set[str] = set()
    if chunk.chunk_id in relevant_chunks:
        keys.add(f"chunk:{chunk.chunk_id}")
    page_key_1 = f"{chunk.filename}:{chunk.page}"
    page_key_2 = str(chunk.page)
    if page_key_1 in relevant_pages:
        keys.add(f"page:{page_key_1}")
    if page_key_2 in relevant_pages:
        keys.add(f"page:{page_key_2}")
    return keys


def _ndcg_at_k(relevance: list[int], ideal_relevant_count: int, k: int) -> float:
    if ideal_relevant_count <= 0:
        return 0.0
    gains = relevance[:k]
    dcg = sum(gain / np.log2(rank + 1) for rank, gain in enumerate(gains, start=1))
    ideal_len = min(k, ideal_relevant_count)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_len + 1))
    return float(dcg / idcg) if idcg else 0.0


def evaluate_retriever(
    index: StudyIndex,
    benchmark: list[dict],
    method: str = "hybrid",
    k: int = 5,
    alpha: float = settings.hybrid_alpha,
    rerank: bool = False,
    candidate_k: int = settings.candidate_k,
    rerank_initial_k: int | None = None,
    rerank_k: int | None = None,
) -> dict[str, float]:
    hits = 0
    reciprocal_ranks: list[float] = []
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    latencies_ms: list[float] = []
    context_words: list[int] = []
    for case in benchmark:
        start = time.perf_counter()
        results = retrieve_for_evaluation(
            index,
            case["question"],
            method=method,
            k=k,
            alpha=alpha,
            rerank=rerank,
            candidate_k=candidate_k,
            rerank_initial_k=rerank_initial_k,
            rerank_k=rerank_k,
        )
        latencies_ms.append((time.perf_counter() - start) * 1000.0)
        context_words.append(sum(len(r.text.split()) for r in results))
        first_rank = 0
        relevance: list[int] = []
        found_keys: set[str] = set()
        for rank, r in enumerate(results, start=1):
            matched_keys = _matched_relevant_keys(r, case)
            new_keys = matched_keys - found_keys
            is_relevant = bool(new_keys)
            relevance.append(int(is_relevant))
            found_keys.update(new_keys)
            if is_relevant and not first_rank:
                first_rank = rank
        hit = first_rank > 0
        hits += int(hit)
        reciprocal_ranks.append(1.0 / first_rank if first_rank else 0.0)
        relevant_keys = _relevant_keys(case)
        relevant_count = len(relevant_keys)
        precisions.append(sum(relevance) / max(1, k))
        recalls.append(len(found_keys) / relevant_count if relevant_count else float(hit))
        ndcgs.append(_ndcg_at_k(relevance, relevant_count or int(hit), k))
    mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else 0.0
    return {
        "method": method,
        "k": k,
        "alpha": alpha,
        "rerank": rerank,
        "candidate_k": candidate_k,
        "rerank_initial_k": rerank_initial_k or 0,
        "rerank_k": rerank_k or 0,
        "num_questions": len(benchmark),
        f"hit@{k}": hits / max(1, len(benchmark)),
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"mrr@{k}": mrr,
        "mrr": mrr,
        "avg_context_words": float(np.mean(context_words)) if context_words else 0.0,
        "latency_ms": float(np.mean(latencies_ms)) if latencies_ms else 0.0,
    }


def evaluate_retriever_by_question_type(
    index: StudyIndex,
    benchmark: list[dict],
    method: str = "hybrid",
    k: int = 5,
    alpha: float = settings.hybrid_alpha,
    rerank: bool = False,
    candidate_k: int = settings.candidate_k,
    rerank_initial_k: int | None = None,
    rerank_k: int | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for question_type in sorted({case.get("question_type", "unknown") for case in benchmark}):
        subset = [case for case in benchmark if case.get("question_type", "unknown") == question_type]
        if not subset:
            continue
        row = evaluate_retriever(
            index,
            subset,
            method=method,
            k=k,
            alpha=alpha,
            rerank=rerank,
            candidate_k=candidate_k,
            rerank_initial_k=rerank_initial_k,
            rerank_k=rerank_k,
        )
        row["question_type"] = question_type
        rows.append(row)
    return pd.DataFrame(rows)


def retrieve_for_evaluation(
    index: StudyIndex,
    query: str,
    method: str = "hybrid",
    k: int = 5,
    alpha: float = settings.hybrid_alpha,
    rerank: bool = False,
    candidate_k: int = settings.candidate_k,
    rerank_initial_k: int | None = None,
    rerank_k: int | None = None,
):
    if not rerank:
        if method == "hybrid":
            return index.retrieve_hybrid(query, k=k, alpha=alpha, candidate_k=candidate_k)
        if method == "rrf":
            return index.retrieve_rrf(query, k=k, candidate_k=candidate_k)
        return index.retrieve(query, method=method, k=k, alpha=alpha, rerank=False)

    initial_k = rerank_initial_k or settings.rerank_initial_k
    final_k = rerank_k or k
    if method == "hybrid":
        initial = index.retrieve_hybrid(query, k=initial_k, alpha=alpha, candidate_k=candidate_k)
    elif method == "rrf":
        initial = index.retrieve_rrf(query, k=initial_k, candidate_k=candidate_k)
    else:
        initial = index.retrieve(query, method=method, k=initial_k, alpha=alpha, rerank=False)
    return index.rerank(query, initial, rerank_k=final_k)


def run_method_comparison(index: StudyIndex, benchmark: list[dict], k: int = 5, alpha: float = settings.hybrid_alpha) -> pd.DataFrame:
    rows = [
        evaluate_retriever(index, benchmark, method="bm25", k=k, alpha=alpha),
        evaluate_retriever(index, benchmark, method="dense", k=k, alpha=alpha),
        evaluate_retriever(index, benchmark, method="hybrid", k=k, alpha=alpha),
        evaluate_retriever(index, benchmark, method="rrf", k=k, alpha=alpha),
    ]
    return pd.DataFrame(rows)


def run_alpha_ablation(index: StudyIndex, benchmark: list[dict], alphas: Iterable[float] = (0.3, 0.5, 0.7), k: int = 5) -> pd.DataFrame:
    rows = [evaluate_retriever(index, benchmark, method="hybrid", k=k, alpha=a) for a in alphas]
    return pd.DataFrame(rows)


def run_topk_ablation(index: StudyIndex, benchmark: list[dict], topks: Iterable[int] = (3, 5, 8), alpha: float = settings.hybrid_alpha) -> pd.DataFrame:
    rows = [evaluate_retriever(index, benchmark, method="hybrid", k=k, alpha=alpha) for k in topks]
    return pd.DataFrame(rows)


def run_chunk_ablation(
    data_dir: Path | str,
    benchmark: list[dict],
    configs: Iterable[tuple[int, int]] = ((300, 50), (450, 80), (700, 100)),
    alpha: float = settings.hybrid_alpha,
    k: int = 5,
    embedding_model: str | None = None,
) -> pd.DataFrame:
    rows: list[dict] = []
    for chunk_size, overlap in configs:
        chunks = build_chunks_from_data_dir(data_dir, chunk_size, overlap)
        index = StudyIndex(embedding_model_name=embedding_model or settings.embedding_model)
        index.build(chunks)
        result = evaluate_retriever(index, benchmark, method="hybrid", k=k, alpha=alpha)
        result.update({"chunk_size": chunk_size, "chunk_overlap": overlap, "num_chunks": len(chunks)})
        rows.append(result)
    return pd.DataFrame(rows)


def citation_marker_validity(answer: str, valid_markers: Iterable[str]) -> dict[str, float]:
    valid = set(valid_markers)
    used = _SOURCE_MARKER_RE.findall(answer)
    invalid = [marker for marker in used if marker not in valid]
    return {
        "has_citation": float(bool(used)),
        "citation_marker_valid": float(bool(used) and not invalid),
        "num_citations": float(len(used)),
        "num_invalid_citations": float(len(invalid)),
    }


def is_refusal(answer: str) -> bool:
    normalized = normalize_answer(answer)
    refusal_patterns = [
        "does not provide enough information",
        "not provide enough information",
        "not enough information",
        "cannot answer",
        "not specified",
        "not mentioned",
        "khong du thong tin",
        "không đủ thông tin",
        "khong duoc cung cap",
        "không được cung cấp",
    ]
    return any(pattern in normalized for pattern in refusal_patterns)


def answerability_accuracy(answer: str, case: dict) -> float:
    answerable = bool(case.get("answerable", case.get("question_type") != "unanswerable"))
    refused = is_refusal(answer)
    return float((answerable and not refused) or ((not answerable) and refused))


def citation_detection_metrics(answer: str, valid_markers: Iterable[str], expected_valid: bool | None = None) -> dict[str, float]:
    valid = set(valid_markers)
    used = _SOURCE_MARKER_RE.findall(answer)
    invalid = [marker for marker in used if marker not in valid]
    has_citation = bool(used)
    citation_valid = has_citation and not invalid
    if expected_valid is None:
        expected_valid = citation_valid
    predicted_invalid = (not has_citation) or bool(invalid)
    actual_invalid = not expected_valid
    return {
        "citation_coverage": float(has_citation),
        "citation_validity": float(citation_valid),
        "invalid_citation_detected": float(predicted_invalid and actual_invalid),
        "missing_citation_detected": float((not has_citation) and actual_invalid),
        "false_positive": float(predicted_invalid and not actual_invalid),
    }


def evaluate_generation_predictions(
    benchmark: list[dict],
    predictions: Iterable[dict],
    answer_key: str = "answer",
    prediction_key: str = "prediction",
    valid_markers_key: str = "valid_markers",
) -> pd.DataFrame:
    """Evaluate already-generated answers without making API calls."""
    reference_by_question = {row["question"]: row.get(answer_key, "") for row in benchmark}
    rows: list[dict] = []
    for item in predictions:
        question = str(item.get("question", "")).strip()
        prediction = str(item.get(prediction_key, "")).strip()
        reference = reference_by_question.get(question, "")
        markers = item.get(valid_markers_key, [])
        marker_metrics = citation_marker_validity(prediction, markers)
        rows.append(
            {
                "question": question,
                "exact_match": exact_match(prediction, reference),
                "answer_f1": token_f1(prediction, reference),
                **marker_metrics,
            }
        )
    return pd.DataFrame(rows)


def save_results(df: pd.DataFrame, path: Path | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
