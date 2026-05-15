import csv
import json
from pathlib import Path

import typer

from src.config import settings
from src.evaluation.chunking_strategies import all_strategies
from src.evaluation.ragas_evaluator import run_evaluation
from src.indexing import ingest
from src.rag import answer
from src.schemas import RagAnswer

app = typer.Typer()


def read_benchmark(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def summary_metrics(df) -> dict[str, float]:
    numeric = df.select_dtypes("number")
    return {col: float(numeric[col].mean()) for col in numeric.columns}


def write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _evaluate_strategy(strategy, output_dir: Path, test_cases: list[dict]) -> dict[str, object]:
    collection_name = f"{settings.qdrant_collection}__{strategy.strategy_id}"
    chunk_count = ingest(recreate=True, collection_name=collection_name, chunker=strategy.chunker)
    result_out: dict[str, object] = {
        "strategy_id": strategy.strategy_id,
        "params": strategy.params,
        "chunk_count": chunk_count,
        "summary_metrics": {},
    }
    try:
        def answer_fn(q: str) -> RagAnswer:
            return answer(q, collection_name=collection_name)

        result = run_evaluation(test_cases, answer_fn=answer_fn, llm_provider="vllm")
        result_out["summary_metrics"] = summary_metrics(result.to_pandas())
    except Exception as exc:
        result_out["error"] = str(exc)
    write_json(output_dir / f"{strategy.strategy_id}.json", result_out)
    return result_out


@app.command()
def main(
    benchmark: Path = Path("src/evaluation/benchmark_rag.csv"),
    output_dir: Path = Path("exports/chunking"),
    include_semantic: bool = False,
):
    cases = read_benchmark(benchmark)
    outputs = [_evaluate_strategy(s, output_dir, cases) for s in all_strategies(include_semantic)]
    write_json(output_dir / "summary.json", {"results": outputs})


if __name__ == "__main__":
    app()
