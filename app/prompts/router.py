"""Prompt used when the router needs an LLM to disambiguate tool choice.

Most routing is done with plain heuristics (app/agent/router.py); this
prompt is a fallback for tasks whose assigned_tool from the planner is
missing or invalid.
"""

SYSTEM = (
    "You select exactly one tool for a research subtask. "
    "Only choose from the tools you are given; never invent a new one."
)


def build_prompt(task_description: str, available_tools: list[str]) -> str:
    return (
        f"Subtask: {task_description}\n"
        f"Available tools: {', '.join(available_tools)}\n"
        "Reply with a JSON object: {\"tool\": \"<one of the available tools>\"}."
    )
