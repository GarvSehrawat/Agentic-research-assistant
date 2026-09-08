"""Document endpoints: upload -> ingest -> embed -> FAISS, listing, search."""
from __future__ import annotations

import logging
import os

from flask import Blueprint, current_app, jsonify, request
from pydantic import BaseModel, Field, ValidationError

from app.models.database import Document, db
from app.retrieval.chunker import chunk_pages
from app.retrieval.loader import DocumentLoadError, load_document
from app.utils.security import FileValidationError, safe_filename, validate_document_file

logger = logging.getLogger(__name__)
document_bp = Blueprint("documents", __name__)


class DocumentSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


@document_bp.post("/documents/upload")
def upload_document():
    if "file" not in request.files:
        return _error("INVALID_REQUEST", "No file part named 'file' in the request.", 400)

    file = request.files["file"]
    if not file or file.filename == "":
        return _error("INVALID_REQUEST", "No file selected.", 400)

    raw = file.read()
    cfg = current_app.config
    try:
        ext = validate_document_file(file.filename, len(raw), cfg["MAX_FILE_SIZE_MB"])
    except FileValidationError as exc:
        return _error("INVALID_FILE", str(exc), 400)

    stored_name = safe_filename(file.filename)
    upload_dir = cfg["UPLOAD_FOLDER"]
    os.makedirs(upload_dir, exist_ok=True)
    stored_path = os.path.join(upload_dir, stored_name)
    with open(stored_path, "wb") as f:
        f.write(raw)

    document = Document(
        filename=stored_name,
        original_filename=file.filename,
        extension=ext,
        size_bytes=len(raw),
    )
    db.session.add(document)
    db.session.commit()

    try:
        pages = load_document(stored_path)
        chunks = chunk_pages(pages, document_id=document.id, filename=file.filename)
        embedding_model = current_app.extensions["embedding_model"]
        vector_store = current_app.extensions["vector_store"]
        if chunks:
            vectors = embedding_model.encode([c.text for c in chunks])
            metadatas = [
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_id": c.chunk_id,
                    "text": c.text,
                }
                for c in chunks
            ]
            vector_store.add_documents(vectors, metadatas)
            vector_store.save()
        document.chunk_count = len(chunks)
        db.session.commit()
    except DocumentLoadError as exc:
        return _error("INGESTION_FAILED", f"Could not process document: {exc}", 422)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Unexpected ingestion failure")
        return _error("INGESTION_FAILED", f"Unexpected error while processing document: {exc}", 500)

    return jsonify(document.to_dict()), 201


@document_bp.get("/documents")
def list_documents():
    docs = Document.query.order_by(Document.created_at.desc()).all()
    return jsonify({"documents": [d.to_dict() for d in docs]}), 200


@document_bp.delete("/documents/<document_id>")
def delete_document(document_id: str):
    document = db.session.get(Document, document_id)
    if document is None:
        return _error("NOT_FOUND", "No document with that id.", 404)

    vector_store = current_app.extensions["vector_store"]
    vector_store.delete_document(document_id)
    vector_store.save()

    upload_path = os.path.join(current_app.config["UPLOAD_FOLDER"], document.filename)
    if os.path.exists(upload_path):
        try:
            os.remove(upload_path)
        except OSError:
            logger.warning("Could not remove uploaded file %s from disk", upload_path)

    db.session.delete(document)
    db.session.commit()
    return jsonify({"deleted": document_id}), 200


@document_bp.post("/documents/search")
def search_documents():
    payload = request.get_json(silent=True)
    if payload is None:
        return _error("INVALID_REQUEST", "Request body must be valid JSON.", 400)
    try:
        req = DocumentSearchRequest.model_validate(payload)
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc.errors()), 400)

    embedding_model = current_app.extensions["embedding_model"]
    vector_store = current_app.extensions["vector_store"]
    if vector_store.size == 0:
        return jsonify({"results": []}), 200

    query_vector = embedding_model.encode([req.query])[0]
    hits = vector_store.search(query_vector, k=req.top_k)
    return jsonify({"results": hits}), 200
