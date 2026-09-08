"""Tool that searches previously-uploaded, embedded documents."""
from __future__ import annotations

from typing import Any

from app.retrieval.embeddings import EmbeddingModel
from app.retrieval.vector_store import VectorStore
from app.schemas.tools import DocumentSearchArgs
from app.tools.base import BaseTool, ToolResult


class DocumentSearchTool(BaseTool):
    name = "document_search"
    description = "Search previously uploaded documents for relevant passages."
    args_schema = DocumentSearchArgs

    def __init__(self, embedding_model: EmbeddingModel, vector_store: VectorStore):
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            args = DocumentSearchArgs.model_validate(kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Invalid document search arguments: {exc}")

        if self.vector_store.size == 0:
            return ToolResult(success=False, error="No documents have been uploaded yet.")

        try:
            query_vec = self.embedding_model.encode([args.query])[0]
            hits = self.vector_store.search(query_vec, k=args.top_k, document_id=args.document_id)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Document search failed: {exc}")

        if not hits:
            return ToolResult(success=True, output="No relevant passages found in the uploaded documents.", sources=[])

        sources = [
            {
                "title": h.get("filename", "document"),
                "document": h.get("filename"),
                "snippet": h.get("text", "")[:500],
                "publisher": f"page {h.get('page_number')}" if h.get("page_number") else None,
            }
            for h in hits
        ]
        output_lines = [f"- ({h.get('filename')}, p.{h.get('page_number')}): {h.get('text', '')[:300]}" for h in hits]
        return ToolResult(success=True, output="\n".join(output_lines), sources=sources)
