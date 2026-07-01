"""ChromaDB vector store creation and loading.

Two entry points, used by strictly different workflows:

- build_vectorstore(): called only by scripts/build_index.py. Creates
  embeddings from documents and persists a fresh Chroma collection.
- load_vectorstore(): called only by the FastAPI app at startup. Opens
  an existing, already-persisted Chroma collection. Never embeds or
  writes new documents.
"""

from __future__ import annotations

from pathlib import Path

from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

from app.utils.logger import get_logger

logger = get_logger(__name__)


def build_vectorstore(
    documents: list[Document],
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: str,
) -> Chroma:
    """Build and persist a new Chroma vector store from documents.

    This is the ONLY function that should create embeddings for storage.
    Used exclusively by the indexing workflow (scripts/build_index.py).

    Args:
        documents: Chunked LangChain Document objects to embed and store.
        embeddings: The embedding model used to vectorize documents.
        collection_name: Name of the Chroma collection to create.
        persist_directory: Directory where Chroma will persist data to disk.

    Returns:
        The populated and persisted Chroma vector store.
    """
    logger.info(
        "Building Chroma vector store: collection=%s, dir=%s, docs=%d",
        collection_name,
        persist_directory,
        len(documents),
    )

    Path(persist_directory).mkdir(parents=True, exist_ok=True)

    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )
    vectordb.persist()

    logger.info("Chroma vector store built and persisted to: %s", persist_directory)
    return vectordb


def load_vectorstore(
    embeddings: HuggingFaceEmbeddings,
    collection_name: str,
    persist_directory: str,
) -> Chroma:
    """Load an existing, already-persisted Chroma vector store.

    This is the ONLY function the FastAPI service should use. It never
    creates embeddings for new documents — it only opens a store that
    scripts/build_index.py has already built.

    Args:
        embeddings: The embedding model matching the one used at build time.
        collection_name: Name of the existing Chroma collection.
        persist_directory: Directory where the Chroma collection was persisted.

    Returns:
        The loaded Chroma vector store, ready for retrieval.

    Raises:
        FileNotFoundError: If the persist directory does not exist.
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        raise FileNotFoundError(
            f"Chroma persist directory not found: {persist_directory}. "
            "Run `python scripts/build_index.py` first."
        )

    logger.info(
        "Loading existing Chroma vector store: collection=%s, dir=%s",
        collection_name,
        persist_directory,
    )

    vectordb = Chroma(
        collection_name=collection_name,
        persist_directory=str(persist_directory),
        embedding_function=embeddings,
    )

    logger.info("Chroma vector store loaded successfully")
    return vectordb


def collection_exists(persist_directory: str, collection_name: str) -> bool:
    """Check whether a non-empty Chroma collection exists on disk.

    Used for startup validation, to fail fast if the API is started
    before the index has been built.

    Args:
        persist_directory: Directory where Chroma persists data.
        collection_name: Name of the collection to check.

    Returns:
        True if the persist directory exists and contains data, False otherwise.
    """
    persist_path = Path(persist_directory)
    if not persist_path.exists():
        return False
    return any(persist_path.iterdir())
