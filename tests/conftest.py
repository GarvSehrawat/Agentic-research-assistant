import re
import shutil
import tempfile

import pytest

from app import create_app
from app.llm.provider import LLMProvider
from app.schemas.research import QueryAnalysis


def _extract_query(prompt: str) -> str:
    """Pull just the user's original query out of a rendered prompt
    template, so heuristics below don't accidentally match boilerplate
    instruction text (which itself contains words like 'current')."""
    match = re.search(r"(?:^|\n)Query:\s*(.+)", prompt)
    if match:
        return match.group(1).strip()
    match = re.search(r"Original query:\s*(.+)", prompt)
    return match.group(1).strip() if match else prompt


def _looks_like_math(text: str) -> bool:
    return bool(re.search(r"\d+\s*[\+\-\*/]\s*\d+", text)) or "calculate" in text.lower()


def _wants_current_info(text: str) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in ("latest", "current", "recent", "today", "news"))


class FakeLLM(LLMProvider):
    """A deterministic stand-in for a real LLM, used throughout the test
    suite so tests never make network calls."""

    def __init__(self):
        self.calls: list[str] = []

    def generate(self, prompt: str, system: str | None = None) -> str:
        self.calls.append(prompt)
        if "Original query" in prompt and "Evidence:" in prompt:
            return "This is a synthesized answer based on the evidence provided. [S1]"
        return "This is a plain factual answer."

    def generate_structured(self, prompt, schema, system=None, max_retries=1):
        self.calls.append(prompt)
        query_text = _extract_query(prompt)

        if schema is QueryAnalysis:
            if _looks_like_math(query_text):
                return QueryAnalysis(
                    query_type="moderate",
                    requires_web_search=False,
                    requires_calculator=True,
                    requires_document_search=False,
                    reason="Contains arithmetic.",
                )
            if _wants_current_info(query_text):
                return QueryAnalysis(
                    query_type="complex",
                    requires_web_search=True,
                    requires_calculator=False,
                    requires_document_search=False,
                    reason="Needs current information.",
                )
            if "document" in query_text.lower() or "report" in query_text.lower():
                return QueryAnalysis(
                    query_type="moderate",
                    requires_web_search=False,
                    requires_calculator=False,
                    requires_document_search=True,
                    reason="Refers to an uploaded document.",
                )
            return QueryAnalysis(
                query_type="simple",
                requires_web_search=False,
                requires_calculator=False,
                requires_document_search=False,
                reason="General knowledge question.",
            )

        # Task list schema (planner) - identified by field name.
        if "tasks" in schema.model_fields:
            if _looks_like_math(query_text):
                tool = "calculator"
            elif _wants_current_info(query_text):
                tool = "web_search"
            elif "document" in query_text.lower() or "report" in query_text.lower():
                tool = "document_search"
            else:
                tool = "none"
            data = {"tasks": [{"description": query_text, "assigned_tool": tool}]}
            return schema.model_validate(data)

        # Sufficiency check schema.
        if "sufficient" in schema.model_fields:
            return schema.model_validate({"sufficient": True, "missing": ""})

        raise AssertionError(f"FakeLLM got an unexpected schema: {schema}")


@pytest.fixture()
def temp_dirs():
    upload_dir = tempfile.mkdtemp()
    faiss_dir = tempfile.mkdtemp()
    yield upload_dir, faiss_dir
    shutil.rmtree(upload_dir, ignore_errors=True)
    shutil.rmtree(faiss_dir, ignore_errors=True)


@pytest.fixture()
def app(temp_dirs):
    upload_dir, faiss_dir = temp_dirs
    application = create_app(
        {
            "TESTING": True,
            "DATABASE_URL": "sqlite:///:memory:",
            "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
            "UPLOAD_FOLDER": upload_dir,
            "FAISS_INDEX_PATH": faiss_dir,
            "LLM_PROVIDER_INSTANCE": FakeLLM(),
            "WEB_SEARCH_API_KEY": None,
            "MAX_AGENT_ITERATIONS": 2,
            "MAX_TOOL_CALLS": 5,
            "MAX_TASKS": 5,
        }
    )
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def fake_llm(app):
    return app.extensions["llm_provider"]
