"""Common tool interface. Only tools registered in the ToolRegistry may
ever be invoked by the agent — there is no arbitrary/dynamic tool
execution path."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    success: bool
    output: Optional[str] = None
    error: Optional[str] = None
    sources: list[dict[str, Any]] = Field(default_factory=list)


class BaseTool(ABC):
    name: str = "base_tool"
    description: str = "Base tool"
    args_schema: type[BaseModel]

    @abstractmethod
    def execute(self, **kwargs: Any) -> ToolResult:
        """Run the tool. Must never raise — catch internally and return
        a ToolResult(success=False, error=...) instead, so the agent
        graph can continue gracefully."""
        raise NotImplementedError


class ToolRegistry:
    """Holds the only tools the agent is allowed to call."""

    def __init__(self) -> None:
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def describe(self) -> list[dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self._tools.values()]
