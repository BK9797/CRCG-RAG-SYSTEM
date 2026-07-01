"""Request schemas for the API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """Request body for the /api/v1/ask endpoint."""

    question: str = Field(
        ...,
        min_length=1,
        description="The natural-language question to ask the knowledge base.",
        examples=["What is the emergency travel assistance number?"],
    )
