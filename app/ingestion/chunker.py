"""Chunking of page-level text into overlapping Document chunks.

Preserves the section-title heuristic and the RecursiveCharacterTextSplitter
configuration from the original Kaggle script, so chunk boundaries and
metadata are identical to the validated implementation.
"""

from __future__ import annotations

import re

from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.utils.logger import get_logger

logger = get_logger(__name__)


def guess_section_title(text: str) -> str:
    """Heuristically guess a section/title label for a page of text.

    Looks at the first ~12 non-empty lines for an explicit "SECTION n:"
    header or an ALL-CAPS-ish heading line. Falls back to the first
    non-empty line, or "Unknown" if the page has no text.

    Args:
        text: Cleaned page text.

    Returns:
        A short string (<=120 chars) representing the guessed section title.
    """
    if not text:
        return "Unknown"

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    for ln in lines[:12]:
        if re.match(r"^SECTION\s+\d+[:\.]", ln, flags=re.I):
            return ln[:120]
        if re.match(r"^[A-Z0-9][A-Z0-9\s&/\-,:\(\)\.]{8,}$", ln) and len(ln) < 120:
            return ln[:120]

    return lines[0][:120] if lines else "Unknown"


def chunk_pages(
    pages: list[dict],
    source_name: str,
    chunk_size: int = 1200,
    chunk_overlap: int = 200,
) -> list[Document]:
    """Split page-level text into overlapping LangChain Document chunks.

    Args:
        pages: List of {"page": int, "text": str} dicts, as produced by
            app.ingestion.pdf_loader.load_pdf_pages.
        source_name: Value stored in each chunk's "source" metadata field
            (typically the original PDF filename).
        chunk_size: Max characters per chunk.
        chunk_overlap: Overlap characters between consecutive chunks.

    Returns:
        A list of Document objects, each with metadata:
        {"source": str, "page": int, "chunk": int, "section": str}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )

    docs: list[Document] = []
    for p in pages:
        page_num = p["page"]
        text = p["text"]
        if not text:
            continue

        section = guess_section_title(text)
        chunks = splitter.split_text(text)
        for idx, chunk in enumerate(chunks):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": source_name,
                        "page": page_num,
                        "chunk": idx,
                        "section": section,
                    },
                )
            )

    logger.info("Loaded %d chunks from %d pages", len(docs), len(pages))
    return docs
