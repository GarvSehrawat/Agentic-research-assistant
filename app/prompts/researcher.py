"""Prompt used when a task needs a plain LLM answer (assigned_tool = none)."""

SYSTEM = (
    "You are a careful research assistant. Answer concisely and factually. "
    "If you are not confident, say so rather than guessing."
)


def build_prompt(task_description: str) -> str:
    return f"Research subtask: {task_description}\n\nProvide a concise, factual answer."
