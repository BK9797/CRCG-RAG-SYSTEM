"""Indexing workflow entry point.

Run this script whenever the source document set needs to be indexed:

    python scripts/build_index.py

This script ONLY builds embeddings and persists them to ChromaDB.
It makes NO calls to the Groq LLM.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly (python scripts/build_index.py)
# by ensuring the project root is on sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config.settings import get_settings  # noqa: E402
from app.embeddings.embedding_model import get_embedding_model  # noqa: E402
from app.ingestion.chunker import chunk_pages  # noqa: E402
from app.ingestion.excel_loader import load_excel_pages  # noqa: E402
from app.ingestion.pdf_loader import load_pdf_pages  # noqa: E402
from app.utils.logger import get_logger  # noqa: E402
from app.vectorstore.chroma_store import build_vectorstore  # noqa: E402

logger = get_logger(__name__)


def main() -> None:
    """Run the full indexing pipeline: load -> clean -> chunk -> embed -> store."""
    logger.info("Loading settings...")
    settings = get_settings()

    data_dir = Path(settings.data_dir)
    if not data_dir.exists():
        logger.error("Dataset directory not found at: %s", data_dir)
        raise FileNotFoundError(
            f"Dataset directory not found at: {data_dir}. Set DATA_DIR in .env or place the files there."
        )

    logger.info("Starting indexing workflow for dataset: %s", data_dir)

    all_pages: list[dict] = []

    pdf_dir = Path(settings.pdf_path)
    if not pdf_dir.is_absolute():
        pdf_dir = pdf_dir if pdf_dir.exists() else data_dir / pdf_dir
    if pdf_dir.exists():
        for pdf_path in sorted(pdf_dir.glob("*.pdf")):
            logger.info("Loading PDF source: %s", pdf_path)
            all_pages.extend(load_pdf_pages(pdf_path))

    excel_dir = Path(settings.excel_path)
    if not excel_dir.is_absolute():
        excel_dir = excel_dir if excel_dir.exists() else data_dir / excel_dir
    if excel_dir.exists():
        for excel_path in sorted(excel_dir.glob("*.xlsx")):
            logger.info("Loading Excel source: %s", excel_path)
            all_pages.extend(load_excel_pages(excel_path))

    if not all_pages:
        logger.error("No text could be extracted from the source dataset. Aborting.")
        raise ValueError("No chunks were produced from the source dataset; nothing to index.")

    documents = chunk_pages(
        pages=all_pages,
        source_name=data_dir.name,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    embeddings = get_embedding_model(settings.embedding_model)

    build_vectorstore(
        documents=documents,
        embeddings=embeddings,
        collection_name=settings.collection_name,
        persist_directory=settings.chroma_db_dir,
    )

    logger.info(
        "Indexing complete. %d chunks stored in collection '%s' at '%s'.",
        len(documents),
        settings.collection_name,
        settings.chroma_db_dir,
    )


if __name__ == "__main__":
    main()
