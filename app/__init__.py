"""Application factory.

Building the LLM provider, embedding model, and tool registry here (once,
at app-creation time) rather than per-request keeps request handling
fast and makes it trivial to inject fakes/mocks in tests via
`create_app(config_overrides=...)`.
"""
from __future__ import annotations

import logging
import os

from flask import Flask

from app.config import Config
from app.models.database import init_db
from app.utils.logging import configure_logging

logger = logging.getLogger(__name__)


def create_app(config_overrides: dict | None = None) -> Flask:
    app = Flask(__name__)
    app.config.from_object(Config)
    if config_overrides:
        app.config.update(config_overrides)

    app.config.setdefault("SQLALCHEMY_DATABASE_URI", app.config["DATABASE_URL"])
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SQLALCHEMY_DATABASE_URI"] = _ensure_sqlite_dir(app.config["SQLALCHEMY_DATABASE_URI"])

    configure_logging("DEBUG" if app.config.get("TESTING") else "INFO")
    init_db(app)

    app.extensions["tool_registry"] = _build_tool_registry(app)
    app.extensions["llm_provider"] = _build_llm_provider(app)

    from app.api.health_routes import health_bp
    from app.api.research_routes import research_bp
    from app.api.document_routes import document_bp

    app.register_blueprint(health_bp, url_prefix="/api")
    app.register_blueprint(research_bp, url_prefix="/api")
    app.register_blueprint(document_bp, url_prefix="/api")

    @app.errorhandler(404)
    def not_found(_):
        return {"error": {"code": "NOT_FOUND", "message": "Resource not found."}}, 404

    @app.errorhandler(500)
    def server_error(exc):
        logger.exception("Unhandled server error")
        return {"error": {"code": "INTERNAL_ERROR", "message": "An internal error occurred."}}, 500

    return app


def _ensure_sqlite_dir(db_uri: str) -> str:
    """Rewrite a relative sqlite:/// URI to an absolute path anchored at
    the project root (instead of Flask-SQLAlchemy's default of
    app.instance_path, which is a separate, less obvious directory) and
    make sure that directory exists. Returns the (possibly rewritten) URI."""
    if not db_uri.startswith("sqlite:///") or db_uri.endswith(":memory:"):
        return db_uri
    path = db_uri.replace("sqlite:///", "", 1)
    if not os.path.isabs(path):
        path = os.path.join(os.getcwd(), path)
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    return f"sqlite:///{path}"


def _build_tool_registry(app: Flask):
    from app.retrieval.embeddings import EmbeddingModel
    from app.retrieval.vector_store import VectorStore
    from app.tools.base import ToolRegistry
    from app.tools.calculator import CalculatorTool
    from app.tools.document_search import DocumentSearchTool
    from app.tools.web_search import WebSearchTool

    registry = ToolRegistry()
    registry.register(CalculatorTool())
    registry.register(
        WebSearchTool(
            api_key=app.config.get("WEB_SEARCH_API_KEY"),
            provider=app.config.get("WEB_SEARCH_PROVIDER", "tavily"),
        )
    )

    embedding_model = app.extensions.get("embedding_model") or EmbeddingModel(app.config.get("EMBEDDING_MODEL"))
    vector_store = app.extensions.get("vector_store") or VectorStore(
        dim=embedding_model.dim, index_path=app.config.get("FAISS_INDEX_PATH")
    )
    app.extensions["embedding_model"] = embedding_model
    app.extensions["vector_store"] = vector_store
    registry.register(DocumentSearchTool(embedding_model=embedding_model, vector_store=vector_store))

    return registry


def _build_llm_provider(app: Flask):
    # Tests can inject a fake provider directly via config to avoid any
    # real network calls.
    injected = app.config.get("LLM_PROVIDER_INSTANCE")
    if injected is not None:
        return injected

    from app.llm.provider import LLMError, get_llm_provider

    try:
        return get_llm_provider(app.config)
    except LLMError as exc:
        logger.warning("LLM provider not available at startup: %s", exc)
        return None
