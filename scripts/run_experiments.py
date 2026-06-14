"""Final StudyAssistant evaluation and ablation runner.

This script implements the final experimental design:

Corpus A:
    Lecture-slide PDFs. Used for the main RAG evaluation and the full ablation
    suite because it is the target application setting.

Corpus B:
    DL_LectureNotes.pdf. Used for robustness evaluation on a long document,
    especially chunking strategy and chunk-size ablations.

Outputs are written under outputs/final_evaluation/ by default.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study_assistant.config import settings
from study_assistant.document import build_chunks_from_pdfs
from study_assistant.evaluation import (
    _is_relevant,
    answerability_accuracy,
    citation_detection_metrics,
    citation_marker_validity,
    evaluate_retriever,
    evaluate_retriever_by_question_type,
    exact_match,
    load_benchmark_csv,
    save_results,
    token_f1,
)
from study_assistant.generation import (
    GeminiClient,
    answer_question,
    generate_flashcards,
    generate_quiz,
    summarize_topic,
)
from study_assistant.retrieval import StudyIndex
from study_assistant.schemas import Chunk


DEFAULT_CHUNK_CONFIGS = [(300, 50), (500, 80), (700, 100), (1000, 150), (1500, 200)]
DEFAULT_CHUNKING_STRATEGIES = ["naive", "paragraph", "semantic"]
DEFAULT_ALPHAS = [round(x / 10, 1) for x in range(11)]
DEFAULT_RERANK_INITIAL_KS = [10, 20, 30, 50]


@dataclass(frozen=True)
class CorpusSpec:
    name: str
    role: str
    pdf_paths: list[Path]
    benchmark_path: Path
    output_dir: Path


def parse_csv_list(value: str, cast=str):
    if not value:
        return []
    return [cast(item.strip()) for item in value.split(",") if item.strip()]


def parse_chunk_configs(value: str) -> list[tuple[int, int]]:
    configs: list[tuple[int, int]] = []
    for item in parse_csv_list(value):
        if ":" in item:
            left, right = item.split(":", 1)
        else:
            left, right = item.split("/", 1)
        configs.append((int(left), int(right)))
    return configs


def ensure_output_dir(path: Path, clean: bool = False) -> Path:
    if clean and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def has_gold_source(case: dict) -> bool:
    return bool(case.get("relevant_pages") or case.get("relevant_chunk_ids"))


def retrieval_cases(benchmark: list[dict]) -> list[dict]:
    return [case for case in benchmark if case.get("answerable", True) and has_gold_source(case)]


def generation_cases(benchmark: list[dict], limit: int = 0) -> list[dict]:
    cases = benchmark[:limit] if limit else benchmark
    return cases


def chunk_summary(chunks: list[Chunk]) -> dict[str, float]:
    lengths = [len(chunk.text.split()) for chunk in chunks]
    return {
        "num_chunks": len(chunks),
        "avg_chunk_length": sum(lengths) / max(1, len(lengths)),
    }


def build_index_from_pdfs(
    pdf_paths: list[Path],
    chunk_size: int,
    overlap: int,
    embedding_model: str,
    bm25_tokenizer: str,
    chunking_strategy: str,
) -> tuple[StudyIndex, list[Chunk], float]:
    start = time.perf_counter()
    chunks = build_chunks_from_pdfs(pdf_paths, chunk_size, overlap, strategy=chunking_strategy)
    index = StudyIndex(embedding_model_name=embedding_model, bm25_tokenizer=bm25_tokenizer)
    index.build(chunks)
    return index, chunks, time.perf_counter() - start


def metric_row(
    index: StudyIndex,
    benchmark: list[dict],
    method: str,
    label: str,
    k: int,
    alpha: float,
    candidate_k: int,
    rerank: bool = False,
    rerank_initial_k: int | None = None,
) -> dict:
    row = evaluate_retriever(
        index,
        benchmark,
        method=method,
        k=k,
        alpha=alpha,
        rerank=rerank,
        candidate_k=candidate_k,
        rerank_initial_k=rerank_initial_k,
        rerank_k=k if rerank else None,
    )
    row["method"] = label
    return row


def direct_gemini_answer(client: GeminiClient, question: str) -> str:
    prompt = f"""
Answer the question concisely.
If the document or context is not available, say that the provided information is insufficient.

Question:
{question}
""".strip()
    return client.generate(prompt).strip()


def aggregate_generation_rows(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .groupby("system")
        .agg(
            em=("exact_match", "mean"),
            token_f1=("answer_f1", "mean"),
            answerability_accuracy=("answerability_accuracy", "mean"),
            citation_coverage=("has_citation", "mean"),
            citation_validity=("citation_marker_valid", "mean"),
            groundedness=("manual_groundedness", "mean"),
            latency_ms=("latency_ms", "mean"),
            num_questions=("question", "count"),
            num_errors=("generation_error", lambda s: int((s.astype(str) != "").sum())),
        )
        .reset_index()
    )


def run_final_retrieval(args: argparse.Namespace, spec: CorpusSpec, index: StudyIndex, benchmark: list[dict]) -> dict[str, pd.DataFrame]:
    cases = retrieval_cases(benchmark)
    row = metric_row(
        index,
        cases,
        "hybrid",
        "hybrid_reranker",
        args.top_k,
        args.alpha,
        max(args.candidate_k, args.rerank_initial_k),
        rerank=True,
        rerank_initial_k=args.rerank_initial_k,
    )
    summary = pd.DataFrame([row])
    save_results(summary, spec.output_dir / "main_final_retrieval.csv")

    by_type = evaluate_retriever_by_question_type(
        index,
        cases,
        method="hybrid",
        k=args.top_k,
        alpha=args.alpha,
        rerank=True,
        candidate_k=max(args.candidate_k, args.rerank_initial_k),
        rerank_initial_k=args.rerank_initial_k,
        rerank_k=args.top_k,
    )
    save_results(by_type, spec.output_dir / "main_final_retrieval_by_type.csv")
    return {"main_final_retrieval": summary, "main_final_retrieval_by_type": by_type}


def run_qa_generation(args: argparse.Namespace, spec: CorpusSpec, index: StudyIndex, benchmark: list[dict]) -> pd.DataFrame:
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for QA generation evaluation.")

    client = GeminiClient(api_key=api_key)
    rows: list[dict] = []
    for case in generation_cases(benchmark, args.generation_limit):
        for system_name in ["direct_gemini", "final_rag_pipeline"]:
            start = time.perf_counter()
            valid_markers: list[str] = []
            try:
                if system_name == "direct_gemini":
                    answer = direct_gemini_answer(client, case["question"])
                else:
                    chunks = index.retrieve_final(
                        case["question"],
                        k=args.top_k,
                        alpha=args.alpha,
                        candidate_k=max(args.candidate_k, args.rerank_initial_k),
                        rerank_initial_k=args.rerank_initial_k,
                    )
                    result = answer_question(case["question"], chunks, client)
                    answer = result.answer
                    valid_markers = [item.source_marker for item in chunks]
                latency_ms = (time.perf_counter() - start) * 1000.0
                citation_metrics = citation_marker_validity(answer, valid_markers)
                generation_error = ""
            except Exception as exc:
                answer = ""
                latency_ms = (time.perf_counter() - start) * 1000.0
                citation_metrics = citation_marker_validity("", valid_markers)
                generation_error = str(exc)

            rows.append(
                {
                    "system": system_name,
                    "question": case["question"],
                    "question_type": case.get("question_type", "unknown"),
                    "reference_answer": case.get("answer", ""),
                    "predicted_answer": answer,
                    "generation_error": generation_error,
                    "exact_match": 0.0 if generation_error else exact_match(answer, case.get("answer", "")),
                    "answer_f1": 0.0 if generation_error else token_f1(answer, case.get("answer", "")),
                    "answerability_accuracy": 0.0 if generation_error else answerability_accuracy(answer, case),
                    "latency_ms": latency_ms,
                    "manual_groundedness": "",
                    **citation_metrics,
                }
            )

    detail = pd.DataFrame(rows)
    save_results(detail, spec.output_dir / "main_qa_generation_details.csv")
    numeric = detail.copy()
    numeric["manual_groundedness"] = pd.to_numeric(numeric["manual_groundedness"], errors="coerce")
    summary = aggregate_generation_rows(numeric.to_dict("records"))
    save_results(summary, spec.output_dir / "main_qa_generation.csv")
    return summary


def run_citation_validation_eval(args: argparse.Namespace, spec: CorpusSpec) -> pd.DataFrame:
    valid_markers = [f"S{i}" for i in range(1, args.top_k + 1)]
    cases = [
        ("valid citation", "The answer is supported [S1].", True),
        ("missing citation", "The answer is supported but has no marker.", False),
        ("invalid citation", "The answer cites a nonexistent marker [S9].", False),
        ("malformed citation", "The answer cites malformed source [source 1].", False),
        ("citation not in context", "The answer cites a marker outside context [S6].", False),
    ]
    rows = []
    for case_type, answer, expected_valid in cases:
        metrics = citation_detection_metrics(answer, valid_markers, expected_valid=expected_valid)
        rows.append(
            {
                "case_type": case_type,
                "detection_rate": metrics["invalid_citation_detected"] or metrics["missing_citation_detected"],
                "false_positive_rate": metrics["false_positive"],
                **metrics,
            }
        )
    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "main_citation_validation.csv")
    return df


def run_learning_output_validation(args: argparse.Namespace, spec: CorpusSpec, index: StudyIndex, benchmark: list[dict]) -> pd.DataFrame:
    api_key = args.google_api_key or os.getenv("GOOGLE_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required for learning output validation.")

    client = GeminiClient(api_key=api_key)
    topics = [case["question"] for case in retrieval_cases(benchmark)[: args.learning_limit]]
    rows: list[dict] = []

    for task in ["summary", "quiz", "flashcards"]:
        json_success = 0
        structure_valid = 0
        duplicates = 0
        citation_valid = 0
        total_items = 0
        errors = 0
        for topic in topics:
            chunks = index.retrieve_final(
                topic,
                k=args.top_k,
                alpha=args.alpha,
                candidate_k=max(args.candidate_k, args.rerank_initial_k),
                rerank_initial_k=args.rerank_initial_k,
            )
            valid_markers = {item.source_marker for item in chunks}
            try:
                if task == "summary":
                    result = summarize_topic(topic, chunks, client=client)
                    json_success += 1
                    total_items += 1
                    structure_valid += int(bool(result.summary and result.key_points))
                    used_markers = {citation.source_marker for citation in result.citations}
                    citation_valid += int(bool(used_markers) and used_markers.issubset(valid_markers))
                elif task == "quiz":
                    result = generate_quiz(topic, chunks, count=args.quiz_count, client=client)
                    json_success += 1
                    questions = [item.question.lower().strip() for item in result.items]
                    total_items += len(result.items)
                    duplicates += len(questions) - len(set(questions))
                    structure_valid += sum(
                        int(len(item.options) == 4 and 0 <= item.correct_index < 4 and bool(item.question))
                        for item in result.items
                    )
                    citation_valid += sum(
                        int(bool(item.source_markers) and set(item.source_markers).issubset(valid_markers))
                        for item in result.items
                    )
                else:
                    result = generate_flashcards(topic, chunks, count=args.flashcard_count, client=client)
                    json_success += 1
                    fronts = [card.front.lower().strip() for card in result.cards]
                    total_items += len(result.cards)
                    duplicates += len(fronts) - len(set(fronts))
                    structure_valid += sum(int(bool(card.front and card.back)) for card in result.cards)
                    citation_valid += sum(
                        int(bool(card.source_markers) and set(card.source_markers).issubset(valid_markers))
                        for card in result.cards
                    )
            except Exception:
                errors += 1

        topic_denom = max(1, len(topics))
        item_denom = max(1, total_items)
        rows.append(
            {
                "task": task,
                "json_success": json_success / topic_denom,
                "structure_validity": structure_valid / item_denom,
                "duplicate_rate": duplicates / item_denom,
                "citation_validity": citation_valid / item_denom,
                "manual_groundedness": "",
                "num_topics": len(topics),
                "num_items": total_items,
                "num_errors": errors,
            }
        )

    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "main_learning_output_validation.csv")
    return df


def run_error_analysis(args: argparse.Namespace, spec: CorpusSpec, index: StudyIndex, benchmark: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    for case in retrieval_cases(benchmark):
        results = index.retrieve_final(
            case["question"],
            k=args.top_k,
            alpha=args.alpha,
            candidate_k=max(args.candidate_k, args.rerank_initial_k),
            rerank_initial_k=args.rerank_initial_k,
        )
        if any(_is_relevant(result, case) for result in results):
            continue
        rows.append(
            {
                "question": case["question"],
                "gold_source": ";".join(case.get("relevant_pages", [])),
                "retrieved_source": ";".join(f"{r.chunk.filename}:{r.chunk.page}" for r in results),
                "generated_answer": "",
                "error_type": "retrieval_failure",
                "cause": "No gold source was retrieved in the final top-k context.",
                "possible_fix": "Improve gold labels, increase candidate_k, tune alpha, inspect chunking, or inspect PDF extraction.",
            }
        )
        if len(rows) >= args.error_limit:
            break

    qa_detail_path = spec.output_dir / "main_qa_generation_details.csv"
    if qa_detail_path.exists() and len(rows) < args.error_limit * 3:
        qa = pd.read_csv(qa_detail_path)
        for _, row in qa.iterrows():
            if str(row.get("generation_error", "")).strip():
                rows.append(
                    {
                        "question": row.get("question", ""),
                        "gold_source": "",
                        "retrieved_source": "",
                        "generated_answer": row.get("predicted_answer", ""),
                        "error_type": "generation_failure",
                        "cause": row.get("generation_error", ""),
                        "possible_fix": "Check Gemini API key, quota, retry delay, and prompt size.",
                    }
                )
            elif float(row.get("citation_marker_valid", 0.0)) == 0.0 and row.get("system") == "final_rag_pipeline":
                rows.append(
                    {
                        "question": row.get("question", ""),
                        "gold_source": "",
                        "retrieved_source": "",
                        "generated_answer": row.get("predicted_answer", ""),
                        "error_type": "citation_failure",
                        "cause": "Generated answer has missing or invalid source markers.",
                        "possible_fix": "Strengthen citation instructions or post-process invalid answers.",
                    }
                )
            if len(rows) >= args.error_limit * 3:
                break

    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "main_error_analysis.csv")
    return df


def run_ablation_chunking_strategy(args: argparse.Namespace, spec: CorpusSpec, benchmark: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    cases = retrieval_cases(benchmark)
    for strategy in args.chunking_strategies:
        chunks = build_chunks_from_pdfs(spec.pdf_paths, args.chunk_size, args.overlap, strategy=strategy)
        index = StudyIndex(embedding_model_name=args.embedding_model, bm25_tokenizer=args.bm25_tokenizer)
        start = time.perf_counter()
        index.build(chunks)
        index_time_sec = time.perf_counter() - start
        row = metric_row(index, cases, "hybrid", strategy, args.top_k, args.alpha, args.candidate_k)
        row.update({"chunking_method": strategy, "index_time_sec": index_time_sec, **chunk_summary(chunks)})
        rows.append(row)
    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "ablation_chunking_strategy.csv")
    return df


def run_ablation_chunk_size(args: argparse.Namespace, spec: CorpusSpec, benchmark: list[dict]) -> pd.DataFrame:
    rows: list[dict] = []
    cases = retrieval_cases(benchmark)
    for chunk_size, overlap in args.chunk_configs:
        chunks = build_chunks_from_pdfs(spec.pdf_paths, chunk_size, overlap, strategy=args.chunking_strategy)
        index = StudyIndex(embedding_model_name=args.embedding_model, bm25_tokenizer=args.bm25_tokenizer)
        start = time.perf_counter()
        index.build(chunks)
        index_time_sec = time.perf_counter() - start
        row = metric_row(index, cases, "hybrid", f"{chunk_size}_{overlap}", args.top_k, args.alpha, args.candidate_k)
        row.update(
            {
                "chunk_size": chunk_size,
                "chunk_overlap": overlap,
                "index_time_sec": index_time_sec,
                **chunk_summary(chunks),
            }
        )
        rows.append(row)
    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "ablation_chunk_size.csv")
    return df


def run_ablation_retrieval_component(
    args: argparse.Namespace,
    spec: CorpusSpec,
    index: StudyIndex,
    benchmark: list[dict],
) -> pd.DataFrame:
    cases = retrieval_cases(benchmark)
    rows = [
        metric_row(index, cases, "bm25", "bm25_only", args.top_k, args.alpha, args.candidate_k),
        metric_row(index, cases, "dense", "dense_only", args.top_k, args.alpha, args.candidate_k),
        metric_row(index, cases, "hybrid", "hybrid_minmax", args.top_k, args.alpha, args.candidate_k),
    ]
    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "ablation_retrieval_component.csv")
    return df


def run_ablation_alpha(args: argparse.Namespace, spec: CorpusSpec, index: StudyIndex, benchmark: list[dict]) -> pd.DataFrame:
    cases = retrieval_cases(benchmark)
    rows = [
        metric_row(index, cases, "hybrid", f"alpha_{alpha:g}", args.top_k, alpha, args.candidate_k)
        for alpha in args.alphas
    ]
    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "ablation_hybrid_alpha.csv")
    return df


def run_ablation_reranker(args: argparse.Namespace, spec: CorpusSpec, index: StudyIndex, benchmark: list[dict]) -> pd.DataFrame:
    cases = retrieval_cases(benchmark)
    rows = [
        metric_row(index, cases, "hybrid", "no_rerank", args.top_k, args.alpha, args.candidate_k)
    ]
    for initial_k in args.rerank_initial_ks:
        rows.append(
            metric_row(
                index,
                cases,
                "hybrid",
                f"rerank_{initial_k}_to_{args.top_k}",
                args.top_k,
                args.alpha,
                max(args.candidate_k, initial_k),
                rerank=True,
                rerank_initial_k=initial_k,
            )
        )
    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "ablation_reranker.csv")
    return df


def _best_row(df: pd.DataFrame, label: str, id_columns: list[str]) -> dict:
    if df.empty:
        return {"selection": label, "reason": "no data"}
    sort_cols = [col for col in ["ndcg@5", "recall@5", "mrr@5", "precision@5"] if col in df.columns]
    best = df.sort_values(sort_cols, ascending=False).iloc[0].to_dict() if sort_cols else df.iloc[0].to_dict()
    selected = {col: best.get(col, "") for col in id_columns}
    selected.update(
        {
            "selection": label,
            "recall@5": best.get("recall@5", ""),
            "mrr@5": best.get("mrr@5", ""),
            "precision@5": best.get("precision@5", ""),
            "ndcg@5": best.get("ndcg@5", ""),
            "latency_ms": best.get("latency_ms", ""),
        }
    )
    return selected


def write_configuration_selection(spec: CorpusSpec, full_ablation: bool) -> pd.DataFrame:
    rows: list[dict] = []
    strategy_path = spec.output_dir / "ablation_chunking_strategy.csv"
    size_path = spec.output_dir / "ablation_chunk_size.csv"
    retrieval_path = spec.output_dir / "ablation_retrieval_component.csv"
    alpha_path = spec.output_dir / "ablation_hybrid_alpha.csv"
    reranker_path = spec.output_dir / "ablation_reranker.csv"

    if strategy_path.exists():
        rows.append(_best_row(pd.read_csv(strategy_path), "best_chunking_strategy", ["chunking_method", "num_chunks", "avg_chunk_length"]))
    if size_path.exists():
        rows.append(_best_row(pd.read_csv(size_path), "best_chunk_size", ["chunk_size", "chunk_overlap", "num_chunks", "avg_context_words", "index_time_sec"]))
    if full_ablation and retrieval_path.exists():
        rows.append(_best_row(pd.read_csv(retrieval_path), "best_retrieval_component", ["method"]))
    if full_ablation and alpha_path.exists():
        rows.append(_best_row(pd.read_csv(alpha_path), "best_alpha", ["alpha", "method"]))
    if full_ablation and reranker_path.exists():
        rows.append(_best_row(pd.read_csv(reranker_path), "best_reranker_setting", ["method", "rerank_initial_k", "rerank_k"]))

    df = pd.DataFrame(rows)
    save_results(df, spec.output_dir / "final_configuration_selection.csv")
    return df


def run_corpus_a(args: argparse.Namespace, spec: CorpusSpec) -> None:
    benchmark = load_benchmark_csv(spec.benchmark_path)
    index, chunks, index_time_sec = build_index_from_pdfs(
        spec.pdf_paths,
        args.chunk_size,
        args.overlap,
        args.embedding_model,
        args.bm25_tokenizer,
        args.chunking_strategy,
    )
    print(f"Corpus A index: {len(chunks)} chunks in {index_time_sec:.2f}s")

    run_final_retrieval(args, spec, index, benchmark)
    run_citation_validation_eval(args, spec)
    if args.run_generation:
        run_qa_generation(args, spec, index, benchmark)
        run_learning_output_validation(args, spec, index, benchmark)
    run_error_analysis(args, spec, index, benchmark)

    # Ablations 1 and 2 are intentionally assigned to Corpus B, where the
    # long-document setting makes chunking effects more informative.
    for stale_name in ["ablation_chunking_strategy.csv", "ablation_chunk_size.csv"]:
        stale_path = spec.output_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    run_ablation_retrieval_component(args, spec, index, benchmark)
    run_ablation_alpha(args, spec, index, benchmark)
    run_ablation_reranker(args, spec, index, benchmark)
    write_configuration_selection(spec, full_ablation=True)
    write_manifest(args, spec, benchmark, chunks, index_time_sec, full_ablation=True)


def run_corpus_b(args: argparse.Namespace, spec: CorpusSpec) -> None:
    benchmark = load_benchmark_csv(spec.benchmark_path)
    index, chunks, index_time_sec = build_index_from_pdfs(
        spec.pdf_paths,
        args.chunk_size,
        args.overlap,
        args.embedding_model,
        args.bm25_tokenizer,
        args.chunking_strategy,
    )
    print(f"Corpus B index: {len(chunks)} chunks in {index_time_sec:.2f}s")

    # Corpus B is a long-document robustness setting. The most informative
    # experiments are chunking strategy and chunk-size sensitivity.
    run_final_retrieval(args, spec, index, benchmark)
    run_ablation_chunking_strategy(args, spec, benchmark)
    run_ablation_chunk_size(args, spec, benchmark)
    write_configuration_selection(spec, full_ablation=False)
    write_manifest(args, spec, benchmark, chunks, index_time_sec, full_ablation=False)


def write_manifest(
    args: argparse.Namespace,
    spec: CorpusSpec,
    benchmark: list[dict],
    chunks: list[Chunk],
    index_time_sec: float,
    full_ablation: bool,
) -> None:
    type_counts = pd.Series([case.get("question_type", "unknown") for case in benchmark]).value_counts().to_dict()
    payload = {
        "corpus": asdict(spec),
        "role": spec.role,
        "num_questions_total": len(benchmark),
        "num_questions_retrieval": len(retrieval_cases(benchmark)),
        "question_type_counts": type_counts,
        "num_chunks_default": len(chunks),
        "avg_chunk_length_default": chunk_summary(chunks)["avg_chunk_length"],
        "index_time_sec_default": index_time_sec,
        "full_ablation_suite": full_ablation,
        "generation_ran": bool(args.run_generation),
        "generation_limit": args.generation_limit,
        "notes": [
            "Retrieval and ablation metrics use answerable questions with gold sources.",
            "Unanswerable questions are reserved for generation answerability evaluation.",
            "Corpus B is used for long-document robustness insight, especially chunking.",
        ],
    }
    payload["corpus"]["pdf_paths"] = [str(path) for path in spec.pdf_paths]
    payload["corpus"]["benchmark_path"] = str(spec.benchmark_path)
    payload["corpus"]["output_dir"] = str(spec.output_dir)
    write_json(spec.output_dir / "manifest.json", payload)


def corpus_specs(args: argparse.Namespace) -> dict[str, CorpusSpec]:
    all_pdfs = sorted(args.data_dir.glob("*.pdf"))
    corpus_a_pdfs = [path for path in all_pdfs if path.name != args.corpus_b_pdf.name]
    corpus_b_pdf = args.data_dir / args.corpus_b_pdf.name
    return {
        "corpusA": CorpusSpec(
            name="corpusA",
            role="main lecture-slide evaluation corpus",
            pdf_paths=corpus_a_pdfs,
            benchmark_path=args.corpus_a_benchmark,
            output_dir=args.output_dir / "corpusA",
        ),
        "corpusB": CorpusSpec(
            name="corpusB",
            role="long-document robustness corpus",
            pdf_paths=[corpus_b_pdf],
            benchmark_path=args.corpus_b_benchmark,
            output_dir=args.output_dir / "corpusB",
        ),
    }


def validate_specs(specs: dict[str, CorpusSpec]) -> None:
    for spec in specs.values():
        missing = [path for path in spec.pdf_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing PDFs for {spec.name}: {missing}")
        if not spec.pdf_paths:
            raise FileNotFoundError(f"No PDFs configured for {spec.name}.")
        if not spec.benchmark_path.exists():
            raise FileNotFoundError(f"Missing benchmark for {spec.name}: {spec.benchmark_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run final StudyAssistant evaluation and ablations.")
    parser.add_argument("command", choices=["all", "corpusA", "corpusB"])
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_evaluation"))
    parser.add_argument("--corpus-a-benchmark", type=Path, default=Path("benchmarks/corpusA_benchmark.csv"))
    parser.add_argument("--corpus-b-benchmark", type=Path, default=Path("benchmarks/corpusB_benchmark.csv"))
    parser.add_argument("--corpus-b-pdf", type=Path, default=Path("DL_LectureNotes.pdf"))
    parser.add_argument("--chunking-strategy", choices=["naive", "paragraph", "semantic"], default="semantic")
    parser.add_argument("--chunk-size", type=int, default=700)
    parser.add_argument("--overlap", type=int, default=100)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--bm25-tokenizer", choices=["regex", "simple", "vi", "vietnamese"], default=settings.bm25_tokenizer)
    parser.add_argument("--candidate-k", type=int, default=30)
    parser.add_argument("--rerank-initial-k", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--alphas", type=lambda x: parse_csv_list(x, float), default=DEFAULT_ALPHAS)
    parser.add_argument("--chunk-configs", type=parse_chunk_configs, default=DEFAULT_CHUNK_CONFIGS)
    parser.add_argument("--chunking-strategies", type=parse_csv_list, default=DEFAULT_CHUNKING_STRATEGIES)
    parser.add_argument("--rerank-initial-ks", type=lambda x: parse_csv_list(x, int), default=DEFAULT_RERANK_INITIAL_KS)
    parser.add_argument("--google-api-key", default="")
    parser.add_argument("--run-generation", action="store_true")
    parser.add_argument("--generation-limit", type=int, default=0)
    parser.add_argument("--learning-limit", type=int, default=10)
    parser.add_argument("--quiz-count", type=int, default=settings.quiz_default_count)
    parser.add_argument("--flashcard-count", type=int, default=settings.flashcard_default_count)
    parser.add_argument("--error-limit", type=int, default=10)
    parser.add_argument("--clean", action="store_true", help="Remove the final output directory before running.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    ensure_output_dir(args.output_dir, clean=args.clean)
    specs = corpus_specs(args)
    validate_specs(specs)

    if args.command in {"all", "corpusA"}:
        ensure_output_dir(specs["corpusA"].output_dir)
        run_corpus_a(args, specs["corpusA"])
    if args.command in {"all", "corpusB"}:
        ensure_output_dir(specs["corpusB"].output_dir)
        run_corpus_b(args, specs["corpusB"])


if __name__ == "__main__":
    main()
