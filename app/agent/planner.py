"""Query analysis and task decomposition.

Both use the LLM's structured-output path so we get validated Pydantic
objects back rather than free text we have to parse ourselves.
"""
from __future__ import annotations

import logging
import uuid

from pydantic import BaseModel, Field

from app.llm.provider import LLMError, LLMProvider
from app.prompts import planner as planner_prompts
from app.prompts import query_analysis as qa_prompts
from app.schemas.research import QueryAnalysis

logger = logging.getLogger(__name__)


class _TaskSpec(BaseModel):
    description: str
    assigned_tool: str = "none"


class _TaskList(BaseModel):
    tasks: list[_TaskSpec] = Field(default_factory=list)


def analyze_query(llm: LLMProvider, query: str) -> QueryAnalysis:
    """Classify the query and flag which tool categories it may need.
    Falls back to a conservative heuristic if the LLM call fails, so a
    single LLM hiccup doesn't kill the whole request."""
    try:
        return llm.generate_structured(
            qa_prompts.build_prompt(query),
            schema=QueryAnalysis,
            system=qa_prompts.SYSTEM,
        )
    except LLMError as exc:
        logger.warning("Query analysis LLM call failed, using heuristic fallback: %s", exc)
        return _heuristic_analysis(query)


def _heuristic_analysis(query: str) -> QueryAnalysis:
    lowered = query.lower()
    has_digits = any(ch.isdigit() for ch in query)
    math_symbols = any(sym in query for sym in ("+", "-", "*", "/", "×", "÷"))
    wants_current = any(word in lowered for word in ("latest", "current", "today", "news", "2025", "2026"))
    return QueryAnalysis(
        query_type="moderate" if (has_digits or wants_current) else "simple",
        requires_web_search=wants_current,
        requires_calculator=has_digits and math_symbols,
        requires_document_search=False,
        reason="Heuristic fallback used because the LLM was unavailable.",
    )


def create_tasks(llm: LLMProvider, query: str, max_tasks: int) -> list[dict]:
    """Decompose a complex query into subtasks. Returns plain dicts
    matching app.agent.state.TaskDict."""
    try:
        result = llm.generate_structured(
            planner_prompts.build_prompt(query, max_tasks),
            schema=_TaskList,
            system=planner_prompts.SYSTEM,
        )
        specs = result.tasks[:max_tasks] or [_TaskSpec(description=query, assigned_tool="none")]
    except LLMError as exc:
        logger.warning("Task decomposition LLM call failed, using single fallback task: %s", exc)
        specs = [_TaskSpec(description=query, assigned_tool="none")]

    return [
        {
            "id": str(uuid.uuid4()),
            "description": spec.description,
            "status": "PENDING",
            "assigned_tool": spec.assigned_tool,
            "result": None,
        }
        for spec in specs
    ]
