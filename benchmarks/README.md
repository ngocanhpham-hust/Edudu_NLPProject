# Benchmark format

Use this folder for manual evaluation data.

Current final-evaluation files:

- `corpusA_benchmark.csv`: lecture-slide benchmark for the main experiments.
- `corpusB_benchmark.csv`: long-document benchmark for `DL_LectureNotes.pdf`.

Required columns:

- `question`: benchmark question
- `answer`: ground-truth answer
- `relevant_pages`: semicolon-separated sources, either `filename:page` or only `page`
- `relevant_chunk_ids`: optional semicolon-separated chunk IDs
- `question_type`: one of `definition`, `factual`, `explanation`, `comparison`, `multi-hop`, `unanswerable`

Example:

```csv
question,answer,relevant_pages,relevant_chunk_ids,question_type
"What is BM25?","BM25 is a lexical retrieval method.","lecture.pdf:5","","definition"
```

If `relevant_chunk_ids` is empty, page-level relevance is used. For
unanswerable questions, set `question_type` to `unanswerable`; the answer can be
a refusal-style reference such as "The document does not provide enough
information." The local Streamlit demo does not need this benchmark file.
