"""Offline Logistic Regression learned fusion ranker for Kaggle/report experiments.

This is intentionally not used as the default live retriever because user-uploaded PDFs have no
relevance labels at upload time.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from .evaluation import _is_relevant
from .retrieval import StudyIndex, keyword_overlap_ratio, minmax
from .config import settings


@dataclass
class LearnedFusionResult:
    model: LogisticRegression
    train_questions: list[str]
    test_questions: list[str]
    metrics: dict[str, float]
    predictions: pd.DataFrame


def _candidate_indices(index: StudyIndex, query: str, candidate_k: int = 30) -> list[int]:
    bm25_scores = index.bm25_scores(query)
    bm25_ids = np.argsort(bm25_scores)[::-1][: min(candidate_k, len(index.chunks))]
    dense_ids, _ = index.dense_scores(query, min(candidate_k, len(index.chunks)))
    hybrid = index.retrieve_hybrid(query, k=min(candidate_k, len(index.chunks)), candidate_k=candidate_k)
    hybrid_ids = []
    id_to_idx = {c.chunk_id: i for i, c in enumerate(index.chunks)}
    for item in hybrid:
        hybrid_ids.append(id_to_idx[item.chunk.chunk_id])
    return sorted(set(map(int, bm25_ids)) | set(map(int, dense_ids)) | set(map(int, hybrid_ids)))


def _features(
    index: StudyIndex,
    query: str,
    idx: int,
    bm25_norm: np.ndarray,
    dense_norm_map: dict[int, float],
    bm25_rank_map: dict[int, int],
    dense_rank_map: dict[int, int],
) -> list[float]:
    chunk = index.chunks[idx]
    length_norm = min(1.0, len(chunk.text.split()) / max(1, settings.chunk_size_words))
    overlap = keyword_overlap_ratio(query, chunk.text)
    bm25_rank_feature = 1.0 / bm25_rank_map[idx] if idx in bm25_rank_map else 0.0
    dense_rank_feature = 1.0 / dense_rank_map[idx] if idx in dense_rank_map else 0.0
    rrf_score = 0.0
    if idx in bm25_rank_map:
        rrf_score += 1.0 / (settings.rrf_k + bm25_rank_map[idx])
    if idx in dense_rank_map:
        rrf_score += 1.0 / (settings.rrf_k + dense_rank_map[idx])
    return [
        float(bm25_norm[idx]),
        float(dense_norm_map.get(idx, 0.0)),
        float(length_norm),
        float(overlap),
        float(bm25_rank_feature),
        float(dense_rank_feature),
        float(rrf_score),
    ]


def build_training_table(index: StudyIndex, benchmark: list[dict], candidate_k: int = 30) -> pd.DataFrame:
    rows: list[dict] = []
    for qid, case in enumerate(benchmark):
        query = case["question"]
        bm25_raw = index.bm25_scores(query)
        bm25_norm = minmax(bm25_raw)
        bm25_order = np.argsort(bm25_raw)[::-1]
        bm25_rank_map = {int(idx): rank for rank, idx in enumerate(bm25_order, start=1)}
        dense_ids, dense_raw = index.dense_scores(query, len(index.chunks))
        dense_norm_raw = minmax(dense_raw)
        dense_norm_map = {int(i): float(s) for i, s in zip(dense_ids, dense_norm_raw)}
        dense_rank_map = {int(idx): rank for rank, idx in enumerate(dense_ids, start=1)}
        for idx in _candidate_indices(index, query, candidate_k=candidate_k):
            dummy = type("Dummy", (), {"chunk": index.chunks[idx]})()
            label = int(_is_relevant(dummy, case))
            f = _features(index, query, idx, bm25_norm, dense_norm_map, bm25_rank_map, dense_rank_map)
            rows.append(
                {
                    "qid": qid,
                    "question": query,
                    "chunk_idx": idx,
                    "chunk_id": index.chunks[idx].chunk_id,
                    "label": label,
                    "bm25_norm": f[0],
                    "dense_norm": f[1],
                    "length_norm": f[2],
                    "overlap": f[3],
                    "bm25_rank_reciprocal": f[4],
                    "dense_rank_reciprocal": f[5],
                    "rrf_score": f[6],
                }
            )
    return pd.DataFrame(rows)


def train_and_evaluate_learned_fusion(
    index: StudyIndex,
    benchmark: list[dict],
    test_size: float = 0.3,
    random_state: int = 42,
    candidate_k: int = 30,
    eval_k: int = 5,
) -> LearnedFusionResult:
    table = build_training_table(index, benchmark, candidate_k=candidate_k)
    qids = sorted(table["qid"].unique().tolist())
    train_qids, test_qids = train_test_split(qids, test_size=test_size, random_state=random_state)
    train = table[table["qid"].isin(train_qids)]
    test = table[table["qid"].isin(test_qids)]
    features = [
        "bm25_norm",
        "dense_norm",
        "length_norm",
        "overlap",
        "bm25_rank_reciprocal",
        "dense_rank_reciprocal",
        "rrf_score",
    ]
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(train[features], train["label"])
    test = test.copy()
    test["prob_relevant"] = model.predict_proba(test[features])[:, 1]

    hits = 0
    rr = []
    for qid, group in test.groupby("qid"):
        ranked = group.sort_values("prob_relevant", ascending=False).head(eval_k)
        labels = ranked["label"].tolist()
        if any(labels):
            hits += 1
            rr.append(1.0 / (labels.index(1) + 1))
        else:
            rr.append(0.0)
    metrics = {
        f"recall@{eval_k}": hits / max(1, len(test_qids)),
        "mrr": float(np.mean(rr)) if rr else 0.0,
        "num_train_questions": len(train_qids),
        "num_test_questions": len(test_qids),
        "num_pairs": len(table),
    }
    for feature, coef in zip(features, model.coef_[0]):
        metrics[f"coef_{feature}"] = float(coef)
    metrics["coef_intercept"] = float(model.intercept_[0])
    return LearnedFusionResult(
        model=model,
        train_questions=[benchmark[i]["question"] for i in train_qids],
        test_questions=[benchmark[i]["question"] for i in test_qids],
        metrics=metrics,
        predictions=test,
    )
