"""Retrieval logic: building a retriever and formatting retrieved docs.

Preserves the MMR (Maximal Marginal Relevance) search configuration from
the original Kaggle script for identical retrieval behavior.
"""

from __future__ import annotations

from langchain.schema import Document
from langchain_community.vectorstores import Chroma
from langchain_core.vectorstores import VectorStoreRetriever

from app.utils.logger import get_logger

logger = get_logger(__name__)


def get_retriever(
    vectordb: Chroma,
    top_k: int = 5,
    fetch_k: int = 12,
) -> VectorStoreRetriever:
    """Create an MMR retriever from a Chroma vector store.

    Args:
        vectordb: The loaded Chroma vector store.
        top_k: Number of chunks to return per query.
        fetch_k: Number of candidates considered before MMR selection.

    Returns:
        A configured VectorStoreRetriever using MMR search.
    """
    logger.info("Creating MMR retriever: k=%d, fetch_k=%d", top_k, fetch_k)
    return vectordb.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": fetch_k},
    )


def retrieve_documents(retriever: VectorStoreRetriever, question: str) -> list[Document]:
    """Retrieve relevant documents for a question.

    Args:
        retriever: A configured VectorStoreRetriever.
        question: The user's question.

    Returns:
        A list of retrieved Document objects.
    """
    docs = retriever.invoke(question)
    logger.info("Retrieved %d documents", len(docs))
    return docs


def format_docs(docs: list[Document]) -> str:
    """Format retrieved documents into a single context string for the prompt.

    Args:
        docs: Retrieved Document objects.

    Returns:
        A string with each document's metadata header and content,
        separated by "---" dividers.
    """
    blocks = []
    for d in docs:
        meta = d.metadata
        header = f"[page {meta.get('page')} | chunk {meta.get('chunk')} | {meta.get('section')}]"
        blocks.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(blocks)


def extract_sources(docs: list[Document]) -> list[dict]:
    """Extract a clean list of source citations from retrieved documents.

    Deduplicates by (page, section) pair while preserving order, so the
    API response doesn't repeat the same page/section multiple times when
    several chunks from the same page were retrieved.

    Args:
        docs: Retrieved Document objects.

    Returns:
        A list of {"page": int, "section": str} dicts.
    """
    seen: set[tuple] = set()
    sources: list[dict] = []
    for d in docs:
        page = d.metadata.get("page")
        section = d.metadata.get("section")
        key = (page, section)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"page": page, "section": section})
    return sources
