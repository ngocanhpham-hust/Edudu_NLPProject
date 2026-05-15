from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel, Field

from src.export import export
from src.filters import MetadataFilter, filters_to_dict
from src.indexing import ingest as ingest_data_dir
from src.indexing import list_documents, save_and_ingest_pdf
from src.learning import generate_flashcards, generate_quiz, summarize as summarize_learning
from src.rag import answer, retrieve
from src.schemas import DocumentInfo, FlashcardSet, QuizSet, RagAnswer, Summary, UploadResponse


class AskRequest(BaseModel):
    question: str = Field(min_length=1)
    k: int | None = Field(default=None, ge=1, le=64)
    filters: MetadataFilter | None = None


class SummarizeRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    k: int | None = Field(default=None, ge=1, le=64)


class QuizRequest(BaseModel):
    document: str | None = None
    query: str | None = None
    filters: MetadataFilter | None = None
    count: int | None = Field(default=None, ge=1, le=50)
    k: int | None = Field(default=None, ge=1, le=64)


class FlashcardsRequest(QuizRequest):
    pass


class ExportRequest(BaseModel):
    kind: str
    payload: dict
    fmt: str = "md"


app = FastAPI(
    title="RAG Learning API",
    description="Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
    version="0.1.0",
)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
def ingest(recreate: bool = False):
    return {"chunks_indexed": ingest_data_dir(recreate=recreate)}


@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    return list_documents()


@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    content = await file.read()
    return save_and_ingest_pdf(content, file.filename or "")


@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    return answer(req.question, k=req.k, filters=filters_to_dict(req.filters))


@app.post("/retrieve")
def retrieve_only(req: AskRequest):
    return retrieve(req.question, k=req.k, filters=filters_to_dict(req.filters))


@app.post("/summarize", response_model=Summary)
def summarize(req: SummarizeRequest):
    return summarize_learning(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        k=req.k,
    )


@app.post("/quiz", response_model=QuizSet)
def quiz(req: QuizRequest):
    return generate_quiz(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )


@app.post("/flashcards", response_model=FlashcardSet)
def flashcards(req: FlashcardsRequest):
    return generate_flashcards(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        count=req.count,
        k=req.k,
    )


@app.post("/export")
def export_payload(req: ExportRequest):
    model_map = {
        "answer": RagAnswer,
        "summary": Summary,
        "quiz": QuizSet,
        "flashcards": FlashcardSet,
    }
    model_class = model_map[req.kind]
    return {"text": export(model_class.model_validate(req.payload), fmt=req.fmt)}
