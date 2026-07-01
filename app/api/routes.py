"""API route definitions.

Routes are intentionally thin: they validate input via Pydantic, delegate
to the RAGService stored in application state, and shape the HTTP response.
No business logic lives here.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.schemas.request import AskRequest
from app.schemas.response import AskResponse, HealthResponse
from app.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["rag"])


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Health check endpoint.

    Verifies that the RAG service started successfully and the vector
    store is loaded.
    """
    rag_service = request.app.state.rag_service
    settings = request.app.state.settings

    logger.info("API request: GET /api/v1/health")

    return HealthResponse(
        status="healthy" if rag_service.is_loaded else "unhealthy",
        vectorstore_loaded=rag_service.is_loaded,
        model=settings.groq_model,
    )


@router.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest, request: Request) -> AskResponse:
    """Answer a question using the RAG pipeline.

    Retrieves relevant chunks from the vector store and generates an
    answer using the Groq LLM, returning the answer with source citations.
    """
    rag_service = request.app.state.rag_service

    logger.info("API request: POST /api/v1/ask")

    if not rag_service.is_loaded:
        raise HTTPException(status_code=503, detail="RAG service is not ready.")

    try:
        return rag_service.ask(payload.question)
    except Exception as exc:  # noqa: BLE001 - surface as a clean 500 to the client
        logger.error("Error answering question: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to generate an answer.") from exc
