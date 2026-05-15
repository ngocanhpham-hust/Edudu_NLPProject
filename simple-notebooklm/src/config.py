from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="RAG_", extra="ignore")

    data_dir: Path = Path("data")
    storage_dir: Path = Path("storage/qdrant")
    qdrant_collection: str = "rag_chunks"

    chunk_size: int = Field(default=1000, ge=100)
    chunk_overlap: int = Field(default=150, ge=0)
    top_k: int = Field(default=5, ge=1, le=64)

    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    embedding_device: str = "cpu"

    llm_provider: Literal["echo", "hf_local", "gemini", "vllm"] = "echo"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)

    hf_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    hf_device: int = -1
    hf_max_new_tokens: int = Field(default=1024, ge=1)

    gemini_model: str = "gemini-2.5-flash"
    google_api_key: str | None = Field(default=None, validation_alias="GOOGLE_API_KEY")

    vllm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    vllm_api_base: str = "http://localhost:8001/v1"
    vllm_api_key: str = "EMPTY"

    summarize_batch_size: int = Field(default=10, ge=1)
    summarize_retrieval_k: int = Field(default=12, ge=1, le=128)
    generation_retrieval_k: int = Field(default=16, ge=1, le=128)
    quiz_default_count: int = Field(default=8, ge=1, le=50)
    flashcards_default_count: int = Field(default=15, ge=1, le=100)

    api_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def validate_config(self) -> "Settings":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size.")
        if self.hf_device < -1:
            raise ValueError("hf_device must be -1 for CPU or >= 0 for CUDA.")
        if self.llm_provider == "gemini" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY is required when llm_provider='gemini'.")
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
