"""Centralized application configuration.

All configurable values are loaded from environment variables (.env file).
This is the single source of truth for configuration across the app —
no module should read os.environ directly; everything goes through Settings.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables / .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Groq / LLM ---
    groq_api_key: str = Field(default="", description="API key for Groq LLM access")
    groq_model: str = Field(
        default="llama-3.3-70b-versatile", description="Groq model identifier"
    )

    # --- Source documents / Ingestion ---
    data_dir: str = Field(
        default="data/CRCG_RAG_Dataset",
        description="Root folder containing the CRCG PDF and Excel source documents",
    )
    pdf_path: str = Field(
        default="data/CRCG_RAG_Dataset/pdf",
        description="Path to the folder containing PDF source documents",
    )
    excel_path: str = Field(
        default="data/CRCG_RAG_Dataset/xlsx",
        description="Path to the folder containing Excel source documents",
    )

    # --- Vector store ---
    chroma_db_dir: str = Field(
        default="data/chroma", description="Directory where Chroma persists its data"
    )
    collection_name: str = Field(
        default="crcg_rag_kb", description="Chroma collection name"
    )

    # --- Embeddings ---
    embedding_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="HuggingFace embedding model identifier",
    )

    # --- Retrieval ---
    top_k: int = Field(default=5, description="Number of chunks to retrieve per query")
    fetch_k: int = Field(
        default=12, description="Number of candidates considered before MMR selection"
    )

    # --- Chunking ---
    chunk_size: int = Field(default=1200, description="Max characters per chunk")
    chunk_overlap: int = Field(default=200, description="Overlap characters between chunks")

    # --- LLM generation ---
    llm_temperature: float = Field(default=0.1, description="Groq sampling temperature")
    llm_max_tokens: int = Field(default=600, description="Max tokens in Groq response")

    # --- API server ---
    api_host: str = Field(default="0.0.0.0", description="Host for the FastAPI server")
    api_port: int = Field(default=8000, description="Port for the FastAPI server")

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Python logging level")

    @field_validator("chroma_db_dir")
    @classmethod
    def _normalize_chroma_dir(cls, v: str) -> str:
        return str(Path(v))

    @property
    def chroma_dir_path(self) -> Path:
        """Chroma directory as a Path object."""
        return Path(self.chroma_db_dir)

    @property
    def pdf_path_obj(self) -> Path:
        """Source document path as a Path object."""
        return Path(self.pdf_path)

    @property
    def excel_path_obj(self) -> Path:
        """Excel source document path as a Path object."""
        return Path(self.excel_path)

    @property
    def data_dir_path(self) -> Path:
        """Dataset root directory as a Path object."""
        return Path(self.data_dir)


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance.

    Using lru_cache ensures the .env file is parsed only once per process
    and the same Settings object is shared across the app.
    """
    return Settings()
