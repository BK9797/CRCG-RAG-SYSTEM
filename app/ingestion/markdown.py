"""Convert cleaned source text into lightweight Markdown before chunking."""

from __future__ import annotations

import re

from app.utils.logger import get_logger

logger = get_logger(__name__)

SECTION_HEADING_PATTERN = re.compile(r"^(SECTION|CHAPTER)\b", flags=re.I)
ALL_CAPS_HEADING_PATTERN = re.compile(
    r"^[A-Z0-9][A-Z0-9\s&/\-,:\(\)\.]{8,}$"
)
BULLET_LINE_PATTERN = re.compile(r"^([*\-•]|[0-9]+\.)\s+")


def convert_page_text_to_markdown(text: str) -> str:
    """Convert a single page's cleaned text into Markdown-like formatting.

    This conversion preserves paragraph structure and promotes obvious
    section titles into Markdown headings, which can improve chunk quality
    and downstream readability.
    """
    if not text:
        return ""

    markdown_lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            markdown_lines.append("")
            continue

        if SECTION_HEADING_PATTERN.match(stripped) or ALL_CAPS_HEADING_PATTERN.match(
            stripped
        ):
            markdown_lines.append(f"## {stripped}")
            continue

        if BULLET_LINE_PATTERN.match(stripped):
            markdown_lines.append(stripped)
            continue

        markdown_lines.append(stripped)

    markdown_text = "\n".join(markdown_lines)
    markdown_text = re.sub(r"\n{3,}", "\n\n", markdown_text)
    return markdown_text.strip()


def convert_pages_to_markdown(pages: list[dict]) -> list[dict]:
    """Convert a list of page dicts into Markdown-enhanced text.

    Args:
        pages: List of {"page": int, "text": str} dicts.

    Returns:
        A new list of dicts with Markdown-formatted page text.
    """
    if not pages:
        return []

    markdown_pages: list[dict] = []
    for page in pages:
        markdown_text = convert_page_text_to_markdown(page.get("text", ""))
        markdown_pages.append({"page": page["page"], "text": markdown_text})

    logger.info("Converted %d pages to Markdown", len(markdown_pages))
    return markdown_pages
