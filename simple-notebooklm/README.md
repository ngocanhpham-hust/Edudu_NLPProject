# Simple NotebookLM

Một project RAG học tập dựa trên PDF, dựng lại theo tài liệu `Building a Simple NotebookLM`.
Hệ thống hỗ trợ hỏi đáp có trích dẫn, tóm tắt, tạo quiz, tạo flashcards, REST API, CLI,
Streamlit UI và scaffold đánh giá Ragas/chunking/reranking.

## Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Mặc định project dùng `RAG_LLM_PROVIDER=echo` để có thể chạy demo không cần API key.
Để sinh câu trả lời thật, đổi sang `gemini`, `vllm` hoặc `hf_local` trong `.env`.

## Nạp dữ liệu

Đặt file PDF vào thư mục `data/`, sau đó:

```bash
python -m src.interfaces.cli ingest --recreate
```

## CLI

```bash
python -m src.interfaces.cli ask "RAG là gì?"
python -m src.interfaces.cli summarize --document ten-file.pdf --fmt md
python -m src.interfaces.cli quiz --query "LoRA" --count 5 --fmt json
python -m src.interfaces.cli flashcards --document ten-file.pdf --count 10
python -m src.interfaces.cli debug-retrieval "chunking là gì?"
```

## API

```bash
uvicorn src.interfaces.api:app --reload
```

Các endpoint chính:

- `GET /health`
- `GET /documents`
- `POST /upload`
- `POST /ask`
- `POST /summarize`
- `POST /quiz`
- `POST /flashcards`

## UI

Chạy API trước, rồi mở Streamlit:

```bash
streamlit run src/interfaces/ui.py
```

## Ghi chú cải tiến

- Có backend `echo` để debug retrieval/prompt khi chưa cấu hình LLM.
- Parser JSON tự bóc code fence và tìm JSON object/array trong output của LLM.
- Validation quiz/flashcard tự bỏ item sai schema, trùng lặp và source marker không tồn tại.
- Reranker là optional, chỉ tải `CrossEncoder` khi chạy thực nghiệm reranking.
