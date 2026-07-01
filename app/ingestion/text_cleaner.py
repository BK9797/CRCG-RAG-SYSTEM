"""Text cleaning utilities for raw PDF-extracted text.

The normalization rules here are preserved exactly from the original
Kaggle implementation to keep retrieval behavior identical.
"""

from __future__ import annotations

import re


def clean_text(text: str) -> str:
    """Normalize PDF-extracted text while preserving meaning.

    Steps:
        1. Remove null bytes and normalize line endings.
        2. Repair hyphenated words broken across line breaks.
        3. Join soft-wrapped lines (single newlines) into flowing text,
           while keeping genuine paragraph breaks (double newlines).
        4. Collapse repeated horizontal whitespace.
        5. Collapse 3+ consecutive blank lines down to a single blank line.

    Args:
        text: Raw text extracted from a PDF page.

    Returns:
        Cleaned, normalized text. Empty string if input is falsy.
    """
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\r", "\n")

    # Fix common line-break hyphenation, e.g. "trav-\nel" -> "travel"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join lines that are wrapped but not real paragraph breaks
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    # Collapse repeated spaces/tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Keep paragraph breaks, collapse many blank lines down to one
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()
