"""Argument schemas for each registered tool."""
from pydantic import BaseModel, Field


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="A basic arithmetic expression, e.g. '25 * 18'")


class WebSearchArgs(BaseModel):
    query: str = Field(..., min_length=1)
    max_results: int = Field(default=5, ge=1, le=10)


class DocumentSearchArgs(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    document_id: str | None = Field(default=None, description="Optional: restrict to one document")
