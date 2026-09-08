"""Pydantic schemas for the research API and agent internals."""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ResearchDepth(str, Enum):
    basic = "basic"
    standard = "standard"
    deep = "deep"


class ResearchStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"


class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    depth: ResearchDepth = ResearchDepth.basic

    @field_validator("query")
    @classmethod
    def query_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("query must not be blank")
        return v.strip()


class SourceRecord(BaseModel):
    source_id: str
    title: str
    url: Optional[str] = None
    document: Optional[str] = None
    publisher: Optional[str] = None
    snippet: Optional[str] = None


class ResearchResponse(BaseModel):
    session_id: str
    query: str
    answer: str = ""
    sources: list[SourceRecord] = Field(default_factory=list)
    status: ResearchStatus = ResearchStatus.completed


class QueryAnalysis(BaseModel):
    """Structured output of the query analyzer."""

    query_type: str = Field(..., description="simple | moderate | complex")
    requires_web_search: bool = False
    requires_calculator: bool = False
    requires_document_search: bool = False
    reason: str = ""


class TaskStatus(str, Enum):
    pending = "PENDING"
    running = "RUNNING"
    completed = "COMPLETED"
    failed = "FAILED"


class ResearchTaskModel(BaseModel):
    id: str
    description: str
    status: TaskStatus = TaskStatus.pending
    assigned_tool: Optional[str] = None
    result: Optional[str] = None
