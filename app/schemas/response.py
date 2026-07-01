"""Response schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    """A single source citation returned alongside an answer."""

    page: int | None = Field(default=None, description="Page number in the source PDF.")
    section: str | None = Field(
        default=None, description="Guessed section/title heading for this page."
    )


class AskResponse(BaseModel):
    """Response body for the /api/v1/ask endpoint."""

    question: str = Field(..., description="The original question that was asked.")
    answer: str = Field(..., description="The generated answer from the RAG pipeline.")
    sources: list[SourceItem] = Field(
        default_factory=list,
        description="List of source pages/sections used to generate the answer.",
    )


class HealthResponse(BaseModel):
    """Response body for the /api/v1/health endpoint."""

    status: str = Field(..., description="Overall service health status.")
    vectorstore_loaded: bool = Field(
        ..., description="Whether the Chroma vector store was loaded successfully."
    )
    model: str = Field(..., description="The Groq model configured for answer generation.")
