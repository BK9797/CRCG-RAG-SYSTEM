"""Source-document loading and page-level text extraction.

Preserves the dual-backend strategy from the original workflow:
PyMuPDF (fitz) is preferred for extraction quality and speed, with pypdf
as a fallback if PyMuPDF is unavailable.
"""

from __future__ import annotations

from pathlib import Path

from app.ingestion.text_cleaner import clean_text
from app.utils.logger import get_logger

logger = get_logger(__name__)

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - exercised only if pymupdf is missing
    fitz = None

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - exercised only if pypdf is missing
    PdfReader = None


def load_pdf_pages(pdf_path: str | Path) -> list[dict]:
    """Extract page-level text from a source document file.

    Tries PyMuPDF first (better extraction fidelity), falling back to
    pypdf if PyMuPDF is not installed.

    Args:
        pdf_path: Path to the source document file on disk.

    Returns:
        A list of dicts, one per page, each shaped as:
        {"page": <1-indexed page number>, "text": <cleaned text>}

    Raises:
        FileNotFoundError: If the source document does not exist at the given path.
        ImportError: If neither PyMuPDF nor pypdf is installed.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")

    logger.info("Loading source document: %s", pdf_path)

    pages: list[dict] = []

    if fitz is not None:
        doc = fitz.open(str(pdf_path))
        try:
            for i in range(len(doc)):
                page = doc[i]
                raw_text = page.get_text("text") or ""
                pages.append({"page": i + 1, "text": clean_text(raw_text)})
        finally:
            doc.close()
        logger.info("Extracted %d pages using PyMuPDF", len(pages))
        return pages

    if PdfReader is not None:
        reader = PdfReader(str(pdf_path))
        for i, page in enumerate(reader.pages):
            raw_text = page.extract_text() or ""
            pages.append({"page": i + 1, "text": clean_text(raw_text)})
        logger.info("Extracted %d pages using pypdf", len(pages))
        return pages

    raise ImportError("Install pymupdf or pypdf to extract text from the source document.")
