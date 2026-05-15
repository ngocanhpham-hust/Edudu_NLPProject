import csv
import json
from pathlib import Path

import typer
from sentence_transformers import CrossEncoder

from src.config import settings
from src.evaluation.ragas_evaluator import run_evaluation
from src.llm import invoke_llm
from src.rag import ANSWER_TEMPLATE, format_citations, render_prompt, retrieve
from src.schemas import RagAnswer

app = typer.Typer()
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"


def answer_with_reranker(
    question: str,
    collection_name: str,
    reranker: CrossEncoder,
    initial_k: int = 15,
    rerank_k: int = 5,
    filters: dict[str, object] | None = None,
) -> RagAnswer:
    chunks = retrieve(question, k=initial_k, filters=filters, collection_name=collection_name)
    if not chunks:
        return RagAnswer(
            question=question,
            answer="Tôi không có đủ thông tin trong ngữ cảnh được cung cấp để trả lời.",
        )
    scores = reranker.predict([[question, chunk.text] for chunk in chunks])
    for chunk, score in zip(chunks, scores, strict=False):
        chunk.score = float(score)
    reranked = sorted(chunks, key=lambda c: c.score, reverse=True)[:rerank_k]
    prompt = render_prompt(ANSWER_TEMPLATE, question=question, chunks=reranked)
    text = invoke_llm(prompt)
    return RagAnswer(
        question=question,
        answer=text.strip(),
        citations=format_citations(reranked),
        chunks=reranked,
    )


def read_benchmark(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


@app.command()
def main(
    benchmark: Path = Path("src/evaluation/benchmark_rag.csv"),
    collection_name: str = settings.qdrant_collection,
    output: Path = Path("exports/reranking/result.json"),
    initial_k: int = 15,
    rerank_k: int = 5,
):
    reranker = CrossEncoder(RERANKER_MODEL)

    def answer_fn(q: str) -> RagAnswer:
        return answer_with_reranker(q, collection_name, reranker, initial_k, rerank_k)

    result = run_evaluation(read_benchmark(benchmark), answer_fn=answer_fn, llm_provider="vllm")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(result.to_pandas().to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    app()
