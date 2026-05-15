import hashlib
import uuid
from collections import defaultdict
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings
from src.schemas import ChunkMetadata, DocumentInfo, UploadResponse
from src.store import ensure_collection, get_vector_store, scroll_all


def discover_pdfs(data_dir: Path | None = None) -> list[Path]:
    root = data_dir or settings.data_dir
    root.mkdir(parents=True, exist_ok=True)
    return sorted(root.glob("*.pdf"))


def _document_id(path: Path) -> str:
    stat = path.stat()
    raw = f"{path.name}:{stat.st_size}:{int(stat.st_mtime)}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def _chunk_id(doc_id: str, page: int, index: int) -> str:
    return f"{doc_id}:{page}:{index}"


def _load_pdf(path: Path) -> list[Document]:
    pages = PyPDFLoader(str(path)).load()
    doc_id = _document_id(path)
    for doc in pages:
        page_number = int(doc.metadata.get("page", 0)) + 1
        doc.metadata = {
            "document_id": doc_id,
            "filename": path.name,
            "source": str(path.resolve()),
            "page": page_number,
            "section": doc.metadata.get("section"),
        }
    return pages


def _splitter(
    chunk_size: int | None = None, chunk_overlap: int | None = None
) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap or settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        keep_separator=False,
    )


def build_chunks(
    pdf_paths: list[Path],
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    chunker: object | None = None,
) -> list[Document]:
    page_docs: list[Document] = []
    for path in pdf_paths:
        page_docs.extend(_load_pdf(path))

    splitter = chunker or _splitter(chunk_size, chunk_overlap)
    chunks = splitter.split_documents(page_docs)
    counters: dict[str, int] = defaultdict(int)

    for chunk in chunks:
        doc_id = chunk.metadata["document_id"]
        idx = counters[doc_id]
        counters[doc_id] += 1
        meta = ChunkMetadata(
            document_id=doc_id,
            filename=chunk.metadata["filename"],
            source=chunk.metadata["source"],
            page=chunk.metadata["page"],
            chunk_id=_chunk_id(doc_id, chunk.metadata["page"], idx),
            section=chunk.metadata.get("section"),
        )
        chunk.metadata = meta.model_dump()
    return chunks


def index_chunks(chunks: list[Document], collection_name: str | None = None) -> int:
    if not chunks:
        return 0
    ids = [str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata["chunk_id"])) for c in chunks]
    get_vector_store(collection_name=collection_name).add_documents(chunks, ids=ids)
    return len(chunks)


def ingest(
    recreate: bool = False,
    collection_name: str | None = None,
    chunker: object | None = None,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> int:
    pdfs = discover_pdfs()
    ensure_collection(recreate=recreate, collection_name=collection_name)
    chunks = build_chunks(
        pdfs, chunker=chunker, chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return index_chunks(chunks, collection_name=collection_name)


def save_and_ingest_pdf(file_bytes: bytes, filename: str) -> UploadResponse:
    safe_name = Path(filename).name
    if not safe_name.lower().endswith(".pdf"):
        raise ValueError("Only PDF files are supported.")
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.data_dir / safe_name
    dest.write_bytes(file_bytes)
    ensure_collection(recreate=False)
    chunks = build_chunks([dest])
    return UploadResponse(filename=safe_name, chunks_indexed=index_chunks(chunks))


def list_documents(collection_name: str | None = None) -> list[DocumentInfo]:
    name = collection_name or settings.qdrant_collection
    docs: dict[str, DocumentInfo] = {}
    for records in scroll_all(name):
        for point in records:
            payload = point.payload or {}
            meta = payload.get("metadata") or {}
            if not meta:
                continue
            doc_id = meta["document_id"]
            info = docs.setdefault(
                doc_id,
                DocumentInfo(
                    document_id=doc_id,
                    filename=meta["filename"],
                    source=meta["source"],
                    pages=[],
                    chunk_count=0,
                ),
            )
            info.chunk_count += 1
            if meta["page"] not in info.pages:
                info.pages.append(meta["page"])
    for info in docs.values():
        info.pages.sort()
    return sorted(docs.values(), key=lambda d: d.filename)
