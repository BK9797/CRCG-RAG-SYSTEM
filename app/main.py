"""FastAPI application entry point.

Handles startup validation and loads the RAG service exactly once,
storing it in application state so it's never recreated per request.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.routes import router
from app.config.settings import get_settings
from app.services.rag_service import RAGService
from app.utils.logger import get_logger
from app.vectorstore.chroma_store import collection_exists

logger = get_logger(__name__)


def validate_startup(settings) -> None:
    """Validate required configuration and resources before startup.

    Checks:
        - GROQ_API_KEY is set
        - The Chroma persist directory exists
        - The Chroma directory is non-empty (i.e. a collection was built)

    Raises:
        RuntimeError: If any validation check fails. The application
        must not start in an invalid state.
    """
    logger.info("Running startup validation...")

    if not settings.groq_api_key:
        msg = "GROQ_API_KEY is not set. Add it to your .env file."
        logger.error(msg)
        raise RuntimeError(msg)

    if not collection_exists(settings.chroma_db_dir, settings.collection_name):
        msg = (
            f"Chroma collection not found at '{settings.chroma_db_dir}'. "
            "Run `python scripts/build_index.py` before starting the API."
        )
        logger.error(msg)
        raise RuntimeError(msg)

    logger.info("Startup validation passed")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: load settings and the RAG service on startup."""
    logger.info("Loading settings...")
    settings = get_settings()

    validate_startup(settings)

    rag_service = RAGService(settings)
    rag_service.load()

    app.state.settings = settings
    app.state.rag_service = rag_service

    logger.info("Application startup complete")
    yield
    logger.info("Application shutting down")


app = FastAPI(
    title="CRCG RAG API",
    description="Retrieval-Augmented Generation service over the CRCG PDF and Excel knowledge base.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(router)
