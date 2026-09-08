"""Splits loaded document pages into reasonably sized, overlapping chunks."""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field

from app.retrieval.loader import PageText


@dataclass
class Chunk:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    text: str


def chunk_pages(
    pages: list[PageText],
    document_id: str,
    filename: str,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    """Chunk by characters (simple and dependency-free) with overlap so
    context isn't lost at chunk boundaries. Keeps chunks a moderate size —
    large enough for context, small enough to stay cheap to embed/search."""
    chunks: list[Chunk] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            piece = text[start:end].strip()
            if piece:
                chunks.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        document_id=document_id,
                        filename=filename,
                        page_number=page.page_number,
                        text=piece,
                    )
                )
            if end == len(text):
                break
            start = end - overlap
    return chunks
