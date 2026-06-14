"""Local CLI for quick testing.

Examples:
    python scripts/local_cli.py build-index --data-dir data
    python scripts/local_cli.py retrieve "What is RAG?" --method final
    python scripts/local_cli.py ask "What is RAG?" --method final
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from study_assistant.config import settings
from study_assistant.generation import GeminiClient, answer_question, result_to_jsonable
from study_assistant.retrieval import StudyIndex


def cmd_build(args):
    index = StudyIndex(embedding_model_name=args.embedding_model, bm25_tokenizer=args.bm25_tokenizer)
    index.build_from_data_dir(args.data_dir, args.chunk_size, args.overlap, chunking_strategy=args.chunking_strategy)
    index.save(args.index_dir)
    print(f"Saved index with {len(index.chunks)} chunks to {args.index_dir}")


def cmd_retrieve(args):
    index = StudyIndex.load(args.index_dir)
    if args.method == "final":
        results = index.retrieve_final(
            args.query,
            k=args.k,
            alpha=args.alpha,
            candidate_k=max(args.candidate_k, args.rerank_initial_k),
            rerank_initial_k=args.rerank_initial_k,
        )
    else:
        results = index.retrieve(args.query, method=args.method, k=args.k, alpha=args.alpha, candidate_k=args.candidate_k, rerank=args.rerank)
    for r in results:
        page_start = r.chunk.page_start or r.chunk.page
        page_end = r.chunk.page_end or r.chunk.page
        page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        print(f"{r.source_marker} score={r.score:.4f} file={r.chunk.filename} {page_label}")
        print(r.text[:500].replace("\n", " "))
        print("-" * 80)


def cmd_ask(args):
    index = StudyIndex.load(args.index_dir)
    if args.method == "final":
        chunks = index.retrieve_final(
            args.question,
            k=args.k,
            alpha=args.alpha,
            candidate_k=max(args.candidate_k, args.rerank_initial_k),
            rerank_initial_k=args.rerank_initial_k,
        )
    else:
        chunks = index.retrieve(args.question, method=args.method, k=args.k, alpha=args.alpha, candidate_k=args.candidate_k, rerank=args.rerank)
    result = answer_question(args.question, chunks, GeminiClient())
    print(result.answer)
    for warning in result.warnings:
        print(f"\nWarning: {warning}")
    print("\nSources:")
    for c in result.citations:
        page_start = c.page_start or c.page
        page_end = c.page_end or c.page
        page_label = f"page {page_start}" if page_start == page_end else f"pages {page_start}-{page_end}"
        print(f"{c.source_marker}: {c.filename}, {page_label}")


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(required=True)

    p = sub.add_parser("build-index")
    p.add_argument("--data-dir", type=Path, default=settings.data_dir)
    p.add_argument("--index-dir", type=Path, default=settings.index_dir)
    p.add_argument("--chunk-size", type=int, default=settings.chunk_size_words)
    p.add_argument("--overlap", type=int, default=settings.chunk_overlap_words)
    p.add_argument("--chunking-strategy", choices=["naive", "paragraph", "semantic"], default="semantic")
    p.add_argument("--embedding-model", default=settings.embedding_model)
    p.add_argument("--bm25-tokenizer", choices=["regex", "simple", "vi", "vietnamese"], default=settings.bm25_tokenizer)
    p.set_defaults(func=cmd_build)

    p = sub.add_parser("retrieve")
    p.add_argument("query")
    p.add_argument("--index-dir", type=Path, default=settings.index_dir)
    p.add_argument("--method", choices=["final", "bm25", "dense", "hybrid", "rrf"], default="final")
    p.add_argument("--k", type=int, default=settings.final_top_k)
    p.add_argument("--alpha", type=float, default=settings.hybrid_alpha)
    p.add_argument("--candidate-k", type=int, default=settings.candidate_k)
    p.add_argument("--rerank-initial-k", type=int, default=settings.rerank_initial_k)
    p.add_argument("--rerank", action="store_true")
    p.set_defaults(func=cmd_retrieve)

    p = sub.add_parser("ask")
    p.add_argument("question")
    p.add_argument("--index-dir", type=Path, default=settings.index_dir)
    p.add_argument("--method", choices=["final", "bm25", "dense", "hybrid", "rrf"], default="final")
    p.add_argument("--k", type=int, default=settings.final_top_k)
    p.add_argument("--alpha", type=float, default=settings.hybrid_alpha)
    p.add_argument("--candidate-k", type=int, default=settings.candidate_k)
    p.add_argument("--rerank-initial-k", type=int, default=settings.rerank_initial_k)
    p.add_argument("--rerank", action="store_true")
    p.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
