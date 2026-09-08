"""Prompt for task decomposition (used for moderate/complex queries)."""

SYSTEM = (
    "You are the planning component of a research assistant. "
    "Break a research query into the smallest number of concrete, "
    "independently-executable subtasks needed to answer it well. "
    "Do not create busy-work tasks."
)


def build_prompt(query: str, max_tasks: int) -> str:
    return (
        f"Query: {query}\n\n"
        f"Produce at most {max_tasks} subtasks needed to answer this query.\n"
        "Each task needs: a short 'description' and an 'assigned_tool' "
        "field which must be one of: web_search, calculator, document_search, none.\n"
        "Return a JSON object with a single key 'tasks', a list of objects "
        "with 'description' and 'assigned_tool'."
    )
