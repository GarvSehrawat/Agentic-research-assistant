"""Document loading for PDF, TXT and Markdown files."""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class PageText:
    page_number: int
    text: str


class DocumentLoadError(Exception):
    pass


def load_document(path: str) -> list[PageText]:
    """Load a document and return a list of (page_number, text).

    Plain-text and Markdown files are treated as a single page.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return _load_pdf(path)
    if ext in (".txt", ".md"):
        return _load_text(path)
    raise DocumentLoadError(f"Unsupported file extension: {ext}")


def _load_text(path: str) -> list[PageText]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        text = f.read()
    return [PageText(page_number=1, text=text)]


def _load_pdf(path: str) -> list[PageText]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover
        raise DocumentLoadError(
            "PyMuPDF is not installed. Run: pip install pymupdf"
        ) from exc

    pages: list[PageText] = []
    try:
        with fitz.open(path) as doc:
            for i, page in enumerate(doc, start=1):
                pages.append(PageText(page_number=i, text=page.get_text()))
    except Exception as exc:  # noqa: BLE001
        raise DocumentLoadError(f"Failed to read PDF: {exc}") from exc
    return pages
