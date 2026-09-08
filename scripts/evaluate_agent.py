#!/usr/bin/env python3
"""Small evaluation harness for the research agent.

Runs a fixed set of representative queries against the agent and
reports metrics that can actually be *measured* from the run - it does
not fabricate scores. Some metrics (task decomposition / tool
selection correctness) are computed against a hand-labeled expected
tool per query; others (average tool calls, citation coverage) are
computed directly from the agent's own execution trace.

Usage:
    OPENAI_API_KEY=... python scripts/evaluate_agent.py
    python scripts/evaluate_agent.py --dry-run   # no LLM required

Metrics reported:
  - tool_selection_accuracy: fraction of queries where the agent's
    chosen tool(s) matched the expected tool category.
  - avg_tool_calls: mean number of tool invocations per query.
  - citation_coverage: fraction of final answers that reference at
    least one source id (e.g. "[S1]") when sources were collected.
  - failure_rate: fraction of runs that ended with status != completed.

A simple-LLM baseline (one direct LLM call, no tools) is also run for
comparison on the same queries, to make the agentic workflow's
tool-usage advantage concrete rather than assumed.
"""
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass, field

sys.path.insert(0, ".")

from app import create_app  # noqa: E402
from app.agent.graph import ResearchAgent  # noqa: E402


@dataclass
class EvalCase:
    query: str
    expected_tool: str  # "calculator" | "web_search" | "document_search" | "none"


EVAL_SET: list[EvalCase] = [
    EvalCase("What is 4567 * 892?", "calculator"),
    EvalCase("What is Python used for?", "none"),
    EvalCase("What are the latest developments in electric vehicles?", "web_search"),
    EvalCase("What is the boiling point of water at sea level?", "none"),
    EvalCase("Calculate 128 / 4 + 10", "calculator"),
]


@dataclass
class RunRecord:
    query: str
    expected_tool: str
    used_tools: list[str] = field(default_factory=list)
    status: str = "unknown"
    answer: str = ""
    tool_calls: int = 0


def run_agentic(agent: ResearchAgent, case: EvalCase) -> RunRecord:
    state = agent.run(query=case.query, depth="basic")
    used_tools = [t.get("assigned_tool") for t in state.get("tasks", []) if t.get("assigned_tool")]
    return RunRecord(
        query=case.query,
        expected_tool=case.expected_tool,
        used_tools=used_tools,
        status=state.get("status", "unknown"),
        answer=state.get("final_answer", ""),
        tool_calls=state.get("tool_calls_made", 0),
    )


def run_baseline(llm, case: EvalCase) -> str:
    """A single direct LLM call with no tools, for comparison."""
    try:
        return llm.generate(case.query)
    except Exception as exc:  # noqa: BLE001
        return f"[baseline failed: {exc}]"


def tool_matches(expected: str, used: list[str]) -> bool:
    if expected == "none":
        return len(used) == 0
    return expected in used


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Skip real LLM calls; just print the eval set.")
    args = parser.parse_args()

    if args.dry_run:
        print("Evaluation set (dry run, no LLM calls made):")
        for case in EVAL_SET:
            print(f"  - '{case.query}' -> expected tool: {case.expected_tool}")
        return

    app = create_app()
    llm = app.extensions.get("llm_provider")
    if llm is None:
        print("No LLM provider configured (set OPENAI_API_KEY). Run with --dry-run instead, or configure a key.")
        sys.exit(1)

    tools = app.extensions["tool_registry"]
    agent = ResearchAgent(
        llm=llm,
        tools=tools,
        max_iterations=app.config["MAX_AGENT_ITERATIONS"],
        max_tool_calls=app.config["MAX_TOOL_CALLS"],
        max_tasks=app.config["MAX_TASKS"],
    )

    records: list[RunRecord] = []
    baselines: dict[str, str] = {}
    with app.app_context():
        for case in EVAL_SET:
            print(f"Running: {case.query}")
            records.append(run_agentic(agent, case))
            baselines[case.query] = run_baseline(llm, case)

    n = len(records)
    correct = sum(1 for r in records if tool_matches(r.expected_tool, r.used_tools))
    avg_tool_calls = sum(r.tool_calls for r in records) / n
    with_sources = [r for r in records if r.used_tools]
    cited = sum(1 for r in with_sources if re.search(r"\[S?\w+\]", r.answer))
    citation_coverage = cited / len(with_sources) if with_sources else float("nan")
    failure_rate = sum(1 for r in records if r.status != "completed") / n

    print("\n=== Results ===")
    print(f"Queries evaluated:        {n}")
    print(f"Tool selection accuracy:  {correct}/{n} = {correct / n:.2%}")
    print(f"Average tool calls/query: {avg_tool_calls:.2f}")
    if with_sources:
        print(f"Citation coverage:        {cited}/{len(with_sources)} = {citation_coverage:.2%}")
    else:
        print("Citation coverage:        n/a (no queries produced sources)")
    print(f"Failure rate:             {failure_rate:.2%}")

    print("\n=== Agentic vs. simple-LLM baseline (qualitative) ===")
    for r in records:
        print(f"\nQuery: {r.query}")
        print(f"  Agent tools used: {r.used_tools or 'none'}")
        print(f"  Agent answer:     {r.answer[:150]}...")
        print(f"  Baseline answer:  {baselines[r.query][:150]}...")

    print(
        "\nNote: the agent's advantage shows up on queries needing arithmetic or "
        "current information (tool use, sourced answers) - not on plain "
        "definitional questions, where a single LLM call is just as good and cheaper."
    )


if __name__ == "__main__":
    main()
