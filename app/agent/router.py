"""Decides which registered tool (if any) should handle a task.

Routing is mostly heuristic — the planner already suggests a tool name,
we just validate it against the registry so nothing unregistered can
ever be invoked. A tiny keyword heuristic acts as a backstop when the
planner's suggestion is missing or invalid.
"""
from __future__ import annotations

import re

from app.tools.base import ToolRegistry

_MATH_RE = re.compile(r"\d+\s*[\+\-\*/%]\s*\d+")


def route_task(description: str, suggested_tool: str | None, registry: ToolRegistry) -> str | None:
    """Return a valid, registered tool name, or None for a plain LLM answer."""
    if suggested_tool and suggested_tool != "none" and registry.get(suggested_tool):
        return suggested_tool

    # Heuristic backstop.
    if _MATH_RE.search(description) and registry.get("calculator"):
        return "calculator"
    lowered = description.lower()
    if any(w in lowered for w in ("latest", "current", "recent", "today", "news")) and registry.get("web_search"):
        return "web_search"
    if any(w in lowered for w in ("uploaded document", "the document", "the report", "the pdf")) and registry.get("document_search"):
        return "document_search"
    return None
