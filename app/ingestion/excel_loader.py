"""Excel workbook loading and tabular text extraction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from app.ingestion.text_cleaner import clean_text
from app.utils.logger import get_logger

logger = get_logger(__name__)


def load_excel_pages(excel_path: str | Path) -> list[dict]:
    """Extract text from an Excel workbook by converting each sheet to a page.

    Args:
        excel_path: Path to the Excel workbook on disk.

    Returns:
        A list of dicts, one per sheet, each shaped as:
        {"page": <1-indexed sheet number>, "text": <cleaned text>}

    Raises:
        FileNotFoundError: If the workbook does not exist at the given path.
    """
    excel_path = Path(excel_path)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel workbook not found at: {excel_path}")

    logger.info("Loading Excel workbook: %s", excel_path)

    pages: list[dict] = []
    excel_file = pd.ExcelFile(excel_path)

    try:
        for sheet_index, sheet_name in enumerate(excel_file.sheet_names, start=1):
            dataframe = pd.read_excel(excel_path, sheet_name=sheet_name)
            if dataframe.empty:
                continue

            rows: list[str] = [f"Sheet: {sheet_name}"]
            for _, row in dataframe.fillna("").iterrows():
                values = [str(value).strip() for value in row.tolist()]
                rows.append(" | ".join(values))

            page_text = clean_text("\n".join(rows))
            if page_text:
                pages.append({"page": sheet_index, "text": page_text})
    finally:
        excel_file.close()

    if not pages:
        logger.warning("No worksheet content could be extracted from %s", excel_path)

    return pages
