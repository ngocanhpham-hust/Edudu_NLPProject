import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from src.export import export
from src.indexing import ingest as ingest_data_dir
from src.indexing import list_documents
from src.learning import generate_flashcards, generate_quiz, summarize as summarize_learning
from src.rag import answer, retrieve

app = typer.Typer(help="Simple NotebookLM CLI")
console = Console()


def _parse_filters(raw: Optional[str]) -> dict | None:
    if not raw:
        return None
    return json.loads(raw)


def _emit(model, output: Optional[Path], fmt: str):
    result = export(model, fmt=fmt, output=output)
    if output is None:
        console.print(result)
    else:
        console.print(f"Wrote {result}")


def _print_sources(chunks):
    table = Table(title="Nguồn")
    table.add_column("Marker")
    table.add_column("File")
    table.add_column("Trang")
    table.add_column("Score")
    for idx, chunk in enumerate(chunks, 1):
        table.add_row(
            f"S{idx}",
            chunk.metadata.filename,
            str(chunk.metadata.page),
            f"{chunk.score:.4f}",
        )
    console.print(table)


@app.command()
def ingest(recreate: bool = typer.Option(False, help="Xóa collection cũ trước khi index.")):
    count = ingest_data_dir(recreate=recreate)
    console.print(f"Done. {count} chunks indexed.")


@app.command()
def documents():
    table = Table(title="Documents")
    table.add_column("Filename")
    table.add_column("Document ID")
    table.add_column("Pages")
    table.add_column("Chunks")
    for doc in list_documents():
        table.add_row(doc.filename, doc.document_id, ",".join(map(str, doc.pages)), str(doc.chunk_count))
    console.print(table)


@app.command()
def ask(question: str, k: Optional[int] = None, filters: Optional[str] = None):
    result = answer(question, k=k, filters=_parse_filters(filters))
    console.print(result.answer)
    _print_sources(result.chunks)


@app.command("debug-retrieval")
def debug_retrieval(
    question: str,
    k: Optional[int] = None,
    filters: Optional[str] = None,
    as_json: bool = False,
):
    chunks = retrieve(question, k=k, filters=_parse_filters(filters))
    if as_json:
        console.print(json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2))
    else:
        _print_sources(chunks)


@app.command("summarize")
def summarize(
    document: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[str] = None,
    k: Optional[int] = None,
    output: Optional[Path] = None,
    fmt: str = "text",
):
    result = summarize_learning(document=document, query=query, filters=_parse_filters(filters), k=k)
    _emit(result, output, fmt)


@app.command()
def quiz(
    document: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[str] = None,
    count: Optional[int] = None,
    k: Optional[int] = None,
    output: Optional[Path] = None,
    fmt: str = "text",
):
    result = generate_quiz(
        document=document, query=query, filters=_parse_filters(filters), count=count, k=k
    )
    _emit(result, output, fmt)


@app.command()
def flashcards(
    document: Optional[str] = None,
    query: Optional[str] = None,
    filters: Optional[str] = None,
    count: Optional[int] = None,
    k: Optional[int] = None,
    output: Optional[Path] = None,
    fmt: str = "text",
):
    result = generate_flashcards(
        document=document, query=query, filters=_parse_filters(filters), count=count, k=k
    )
    _emit(result, output, fmt)


if __name__ == "__main__":
    app()
