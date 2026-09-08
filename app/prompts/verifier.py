"""Prompt for the evidence-sufficiency check used in iterative research."""

SYSTEM = (
    "You judge whether the collected evidence is sufficient to answer the "
    "user's query. Be strict: if key facts are missing, say so."
)


def build_prompt(query: str, evidence_summary: str) -> str:
    return (
        f"Original query: {query}\n\n"
        f"Evidence gathered so far:\n{evidence_summary}\n\n"
        "Is this evidence sufficient to answer the query well? "
        "Reply with a JSON object: "
        "{\"sufficient\": true|false, \"missing\": \"<what's missing, or empty string>\"}."
    )
