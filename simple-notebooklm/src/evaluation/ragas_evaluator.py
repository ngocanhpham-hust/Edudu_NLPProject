from collections.abc import Callable

from datasets import Dataset
from ragas import evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

from src.llm import get_llm
from src.rag import answer as default_answer_fn
from src.schemas import RagAnswer
from src.store import get_embeddings


def get_ragas_metrics(llm, embeddings):
    faithfulness.llm = llm
    answer_relevancy.llm = llm
    context_precision.llm = llm
    context_precision.embeddings = embeddings
    context_recall.llm = llm
    context_recall.embeddings = embeddings
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def run_evaluation(
    test_cases: list[dict[str, str]],
    *,
    answer_fn: Callable[[str], RagAnswer] = default_answer_fn,
    llm_provider: str | None = None,
    timeout_s: int = 180,
    max_retries: int = 3,
    max_workers: int = 4,
):
    data: dict[str, list] = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }
    for case in test_cases:
        rag_response = answer_fn(case["question"])
        data["user_input"].append(case["question"])
        data["response"].append(rag_response.answer)
        data["retrieved_contexts"].append([chunk.text for chunk in rag_response.chunks])
        data["reference"].append(case["ground_truth"])

    dataset = Dataset.from_dict(data)
    llm = LangchainLLMWrapper(get_llm(provider=llm_provider))
    embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    metrics = get_ragas_metrics(llm, embeddings)
    config = RunConfig(timeout=timeout_s, max_retries=max_retries, max_workers=max_workers)
    return evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings, run_config=config)
