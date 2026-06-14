# Lightweight NotebookLM-inspired NLP Study Assistant

A final-source-code layout for an NLP course project.

## Project idea

User uploads PDF learning documents and studies with them through:

- document-grounded Q&A
- information extraction
- summarization
- quiz generation
- flashcard generation
- citations by filename/page/chunk

Core retrieval pipeline:

```text
PDF -> page text extraction -> structure-preserving cleaning -> paragraph/sentence-aware chunking
    -> BM25 + SentenceTransformer embeddings -> FAISS exact inner-product search
    -> hybrid min-max fusion or Reciprocal Rank Fusion -> optional reranking -> Gemini generation
```

Implemented method improvements:

- Stable `document_id` from SHA1 file bytes, so benchmark labels are not tied to file mtime.
- Paragraph/sentence-aware chunking that preserves useful line breaks from slides, formulas, bullets, and tables.
- Optional Vietnamese BM25 tokenization via `BM25_TOKENIZER=vi` when `underthesea` or `pyvi` is installed; otherwise regex tokenization is used.
- E5 embedding models are encoded with `query:` and `passage:` prefixes automatically.
- Hybrid default alpha is `0.70` in the final evaluation runner.
- RRF retrieval is available as a scale-free fusion baseline.
- Generated citations are filtered to source markers actually used in the answer; invalid or missing markers are surfaced as warnings.
- Summarization supports both retrieval-based topic summary and map-reduce full-document summary.

## Repository structure

```text
final-notebooklm-study-assistant/
  app.py                              # LOCAL Streamlit demo
  scripts/local_cli.py                # LOCAL CLI test tool
  scripts/run_experiments.py          # Final Corpus A/B evaluation runner
  src/study_assistant/                # Shared core code
    config.py
    schemas.py
    document.py                       # PDF parsing + cleaning + chunking
    retrieval.py                      # BM25 + dense FAISS + hybrid/RRF + reranker
    generation.py                     # Gemini prompts + answer/summary/quiz/flashcards
    evaluation.py                     # Recall@k, MRR, ablations
    learned_fusion.py                 # Offline Logistic Regression fusion ranker
  benchmarks/
    corpusA_benchmark.csv             # Lecture-slide benchmark with question_type
    corpusB_benchmark.csv             # DL_LectureNotes long-document benchmark
  data/                               # Put local PDFs here
  outputs/                            # Experiment/generation outputs
  indexes/                            # Saved FAISS index for local CLI
  requirements.txt
  .env.example
```

## Local demo

```bash
cd final-notebooklm-study-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export GOOGLE_API_KEY="your_key"
streamlit run app.py
```

The local app is for the final presentation/demo.

## Local CLI

```bash
python scripts/local_cli.py build-index --data-dir data
python scripts/local_cli.py retrieve "What is RAG?" --method rrf
python scripts/local_cli.py ask "What is RAG?" --method hybrid
```

For Vietnamese BM25 ablation, install `underthesea` or `pyvi`, then build with:

```bash
python scripts/local_cli.py build-index --data-dir data --bm25-tokenizer vi
```

## Which files run where?

| Environment | Run these files | Purpose |
|---|---|---|
| Local | `app.py` | Streamlit final demo |
| Local | `scripts/local_cli.py` | Quick command-line testing |
| Local/Kaggle | `scripts/run_experiments.py` | Final evaluation and ablations |
| Both | `src/study_assistant/*.py` | Shared core code |

## Benchmark note

The final evaluation uses two benchmark files:

- `benchmarks/corpusA_benchmark.csv`: lecture-slide corpus with 100 answerable questions and 20 unanswerable questions.
- `benchmarks/corpusB_benchmark.csv`: long-document benchmark for `DL_LectureNotes.pdf`.

Retrieval evaluation reports `Hit@k`, `Precision@k`, `Recall@k`, `nDCG@k`, and `MRR`. Generation utilities include exact match, token F1, and citation marker validity for already-generated predictions.

## Experiment runner

The final protocol is implemented in `scripts/run_experiments.py`.

Run the full retrieval/citation/error-analysis/ablation suite for both corpora:

```bash
python scripts/run_experiments.py all --clean
```

Run only the main lecture-slide corpus:

```bash
python scripts/run_experiments.py corpusA --clean
```

Run only the long-document robustness corpus:

```bash
python scripts/run_experiments.py corpusB --clean
```

Generation-side experiments call Gemini and require an API key:

```bash
python scripts/run_experiments.py corpusA \
  --run-generation \
  --generation-limit 30 \
  --google-api-key "$GOOGLE_API_KEY"
```

Outputs are written under `outputs/final_evaluation/corpusA/` and
`outputs/final_evaluation/corpusB/`.
