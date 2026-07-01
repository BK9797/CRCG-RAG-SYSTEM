"""HuggingFace embedding model factory.

Centralizes embedding model creation so the same configuration is used
consistently during both indexing (build_index.py) and serving (FastAPI).
"""

from __future__ import annotations

from langchain_community.embeddings import HuggingFaceEmbeddings

from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_embedding_model(model_name: str) -> HuggingFaceEmbeddings:
    """Create a HuggingFace embeddings instance.

    Args:
        model_name: HuggingFace model identifier, e.g. "BAAI/bge-small-en-v1.5".

    Returns:
        A configured HuggingFaceEmbeddings instance running on CPU with
        normalized embeddings (matches the original Kaggle configuration).
    """
    logger.info("Loading embedding model: %s", model_name)
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
