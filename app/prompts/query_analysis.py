"""Prompt for the query-analysis step."""

SYSTEM = (
    "You are the query-analysis component of a research assistant. "
    "Classify the user's query and decide which tools it plausibly needs. "
    "Be conservative: only request a tool if it is genuinely useful."
)


def build_prompt(query: str) -> str:
    return (
        "Analyze this research query and classify it.\n\n"
        f"Query: {query}\n\n"
        "query_type must be one of: simple, moderate, complex.\n"
        "- simple: a definitional or conceptual question answerable from general knowledge.\n"
        "- moderate: needs one lookup or calculation.\n"
        "- complex: needs multiple steps, comparisons, or several sources.\n"
        "Set requires_web_search=true only if current/external information is needed.\n"
        "Set requires_calculator=true only if arithmetic is needed.\n"
        "Set requires_document_search=true only if the query refers to an uploaded document.\n"
        "Give a one-sentence reason."
    )
