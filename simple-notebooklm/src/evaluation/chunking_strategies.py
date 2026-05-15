from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.store import get_embeddings

DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]
_RECURSIVE_CONFIGS = [
    ("rc_500_50", 500, 50),
    ("rc_800_100", 800, 100),
    ("rc_1000_150", 1000, 150),
    ("rc_1500_200", 1500, 200),
]
_SEMANTIC_CONFIGS = [
    ("semantic_percentile", "percentile"),
    ("semantic_std_dev", "standard_deviation"),
    ("semantic_interquartile", "interquartile"),
]


@dataclass(frozen=True)
class ChunkingStrategy:
    strategy_id: str
    chunker: object
    params: dict[str, object]


@dataclass(frozen=True)
class RecursiveChunker:
    chunk_size: int = 1000
    chunk_overlap: int = 150
    separators: list[str] | None = None

    def _splitter(self) -> RecursiveCharacterTextSplitter:
        return RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators or DEFAULT_SEPARATORS,
            is_separator_regex=False,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [] if not documents else self._splitter().split_documents(documents)


@dataclass(frozen=True)
class SemanticChunkerWrapper:
    breakpoint_type: str = "percentile"

    def _splitter(self):
        from langchain_experimental.text_splitter import SemanticChunker

        return SemanticChunker(
            embeddings=get_embeddings(),
            breakpoint_threshold_type=self.breakpoint_type,
        )

    def split_documents(self, documents: list[Document]) -> list[Document]:
        return [] if not documents else self._splitter().split_documents(documents)

    def split_text(self, text: str) -> list[str]:
        return self._splitter().split_text(text)


def recursive_strategies() -> list[ChunkingStrategy]:
    return [
        ChunkingStrategy(sid, RecursiveChunker(size, overlap), {"size": size, "overlap": overlap})
        for sid, size, overlap in _RECURSIVE_CONFIGS
    ]


def semantic_strategies() -> list[ChunkingStrategy]:
    return [
        ChunkingStrategy(sid, SemanticChunkerWrapper(kind), {"breakpoint_type": kind})
        for sid, kind in _SEMANTIC_CONFIGS
    ]


def all_strategies(include_semantic: bool = False) -> list[ChunkingStrategy]:
    strategies = recursive_strategies()
    if include_semantic:
        strategies.extend(semantic_strategies())
    return strategies
