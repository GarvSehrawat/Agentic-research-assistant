"""Web search tool.

Provider is configurable via WEB_SEARCH_PROVIDER / WEB_SEARCH_API_KEY.
Currently supports Tavily (a search API built for LLM agents) because it
returns clean, structured results with minimal setup. If no API key is
configured, the tool fails gracefully and explains why — it never
fabricates results.
"""
from __future__ import annotations

import logging
from typing import Any

import requests

from app.schemas.tools import WebSearchArgs
from app.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

TAVILY_ENDPOINT = "https://api.tavily.com/search"


class WebSearchTool(BaseTool):
    name = "web_search"
    description = "Search the public web for current information and return titled, sourced snippets."
    args_schema = WebSearchArgs

    def __init__(self, api_key: str | None = None, provider: str = "tavily", timeout: int = 10):
        self.api_key = api_key
        self.provider = provider
        self.timeout = timeout

    def execute(self, **kwargs: Any) -> ToolResult:
        try:
            args = WebSearchArgs.model_validate(kwargs)
        except Exception as exc:  # noqa: BLE001
            return ToolResult(success=False, error=f"Invalid web search arguments: {exc}")

        if not self.api_key:
            return ToolResult(
                success=False,
                error=(
                    "Web search is unavailable: no WEB_SEARCH_API_KEY is configured. "
                    "Set WEB_SEARCH_API_KEY in your .env to enable it."
                ),
            )

        if self.provider != "tavily":
            return ToolResult(success=False, error=f"Unsupported web search provider: {self.provider}")

        try:
            response = requests.post(
                TAVILY_ENDPOINT,
                json={
                    "api_key": self.api_key,
                    "query": args.query,
                    "max_results": args.max_results,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.warning("Web search request failed: %s", exc)
            return ToolResult(success=False, error=f"Web search request failed: {exc}")
        except ValueError as exc:
            return ToolResult(success=False, error=f"Web search returned malformed JSON: {exc}")

        results = data.get("results", [])
        if not results:
            return ToolResult(success=True, output="No web results found.", sources=[])

        sources = [
            {
                "title": r.get("title", "Untitled"),
                "url": r.get("url"),
                "snippet": (r.get("content") or "")[:500],
                "publisher": _domain(r.get("url")),
            }
            for r in results
        ]
        output_lines = [f"- {s['title']}: {s['snippet']}" for s in sources]
        return ToolResult(success=True, output="\n".join(output_lines), sources=sources)


def _domain(url: str | None) -> str | None:
    if not url:
        return None
    try:
        return url.split("/")[2]
    except IndexError:
        return None
