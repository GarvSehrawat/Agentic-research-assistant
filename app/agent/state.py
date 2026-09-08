"""Shared agent state.

This is a plain TypedDict (not a Pydantic model) because LangGraph's
StateGraph expects a mapping it can merge partial updates into between
nodes. Pydantic schemas (app/schemas) are used at the boundaries
(API request/response, structured LLM output) instead.
"""
from __future__ import annotations

from typing import Any, TypedDict


class TaskDict(TypedDict, total=False):
    id: str
    description: str
    status: str
    assigned_tool: str | None
    result: str | None


class SourceDict(TypedDict, total=False):
    source_id: str
    title: str
    url: str | None
    document: str | None
    publisher: str | None
    snippet: str | None


class AgentState(TypedDict, total=False):
    # inputs
    query: str
    depth: str
    document_id: str | None

    # working memory
    analysis: dict[str, Any]
    tasks: list[TaskDict]
    tool_results: list[dict[str, Any]]
    sources: list[SourceDict]

    # control
    iteration: int
    tool_calls_made: int
    evidence_sufficient: bool

    # outputs
    final_answer: str
    errors: list[str]
    status: str


def new_state(query: str, depth: str = "basic", document_id: str | None = None) -> AgentState:
    return AgentState(
        query=query,
        depth=depth,
        document_id=document_id,
        analysis={},
        tasks=[],
        tool_results=[],
        sources=[],
        iteration=0,
        tool_calls_made=0,
        evidence_sufficient=False,
        final_answer="",
        errors=[],
        status="pending",
    )
