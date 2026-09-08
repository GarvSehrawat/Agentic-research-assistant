"""The agentic workflow.

    START -> analyze_query -> create_plan -> execute_tools -> check_evidence
                                                      ^              |
                                                      |   NO (loop)  |
                                                      +--------------+
                                                                     | YES
                                                                     v
                                                          synthesize_answer -> END

Built with LangGraph when it's installed (it's listed as a required
dependency); falls back to an equivalent hand-rolled sequential runner
if LangGraph is unavailable, so the app still degrades gracefully
rather than crashing on import.

Simple queries skip planning/tools entirely and go straight to
synthesis with the LLM's own knowledge - we don't manufacture busywork
tasks for "What is Python?".
"""
from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Callable

from app.agent import evidence as evidence_mod
from app.agent.planner import analyze_query, create_tasks
from app.agent.router import route_task
from app.agent.state import AgentState, new_state
from app.llm.provider import LLMError, LLMProvider
from app.prompts import researcher as researcher_prompts
from app.prompts import synthesizer as synthesizer_prompts
from app.prompts import verifier as verifier_prompts
from app.tools.base import ToolRegistry
from app.utils.security import fence_untrusted_content

logger = logging.getLogger(__name__)

try:
    from langgraph.graph import END, START, StateGraph

    _HAS_LANGGRAPH = True
except ImportError:  # pragma: no cover
    _HAS_LANGGRAPH = False


class _SufficiencyCheck:
    def __init__(self, sufficient: bool, missing: str = ""):
        self.sufficient = sufficient
        self.missing = missing


class ResearchAgent:
    """Wraps the compiled graph (or fallback runner) plus its dependencies."""

    def __init__(
        self,
        llm: LLMProvider,
        tools: ToolRegistry,
        max_iterations: int = 3,
        max_tool_calls: int = 20,
        max_tasks: int = 15,
    ):
        self.llm = llm
        self.tools = tools
        self.max_iterations = max_iterations
        self.max_tool_calls = max_tool_calls
        self.max_tasks = max_tasks
        self._runner = self._build_runner()

    # ---- graph nodes -----------------------------------------------------

    def _node_analyze_query(self, state: AgentState) -> dict:
        analysis = analyze_query(self.llm, state["query"])
        return {"analysis": analysis.model_dump()}

    def _node_create_plan(self, state: AgentState) -> dict:
        analysis = state["analysis"]
        if analysis.get("query_type") == "simple":
            # No busywork for simple questions.
            return create_tasks(self.llm, state["query"], max_tasks=1)

        depth_task_caps = {"basic": 3, "standard": 6, "deep": self.max_tasks}
        cap = min(depth_task_caps.get(state.get("depth", "basic"), 3), self.max_tasks)
        tasks = create_tasks(self.llm, state["query"], max_tasks=cap)
        return {"tasks": tasks}

    def _node_execute_tools(self, state: AgentState) -> dict:
        tasks = list(state.get("tasks", []))
        tool_results = list(state.get("tool_results", []))
        sources = list(state.get("sources", []))
        errors = list(state.get("errors", []))
        calls_made = state.get("tool_calls_made", 0)

        for task in tasks:
            if task.get("status") == "COMPLETED":
                continue
            if calls_made >= self.max_tool_calls:
                task["status"] = "FAILED"
                task["result"] = "Skipped: MAX_TOOL_CALLS limit reached."
                errors.append(f"Task '{task['id']}' skipped: tool call limit reached.")
                continue

            tool_name = route_task(task["description"], task.get("assigned_tool"), self.tools)
            task["status"] = "RUNNING"

            if tool_name is None:
                # Plain LLM answer, no tool involved.
                try:
                    answer = self.llm.generate(
                        researcher_prompts.build_prompt(task["description"]),
                        system=researcher_prompts.SYSTEM,
                    )
                    task["status"] = "COMPLETED"
                    task["result"] = answer
                    tool_results.append({"task_id": task["id"], "tool": "llm", "success": True})
                except LLMError as exc:
                    task["status"] = "FAILED"
                    task["result"] = None
                    errors.append(str(exc))
                    tool_results.append({"task_id": task["id"], "tool": "llm", "success": False, "error": str(exc)})
                continue

            tool = self.tools.get(tool_name)
            calls_made += 1
            try:
                if tool_name == "calculator":
                    result = tool.execute(expression=_extract_expression(task["description"]))
                elif tool_name == "web_search":
                    result = tool.execute(query=task["description"])
                elif tool_name == "document_search":
                    result = tool.execute(query=task["description"], document_id=state.get("document_id"))
                else:
                    result = tool.execute(query=task["description"])
            except Exception as exc:  # noqa: BLE001 - tools shouldn't raise, but stay safe
                task["status"] = "FAILED"
                errors.append(f"Tool '{tool_name}' raised an unexpected error: {exc}")
                tool_results.append({"task_id": task["id"], "tool": tool_name, "success": False, "error": str(exc)})
                continue

            task["assigned_tool"] = tool_name
            if result.success:
                task["status"] = "COMPLETED"
                task["result"] = result.output
                for src in result.sources:
                    sources.append({"source_id": str(uuid.uuid4())[:8], **src})
            else:
                task["status"] = "FAILED"
                task["result"] = None
                errors.append(f"Tool '{tool_name}' failed: {result.error}")
            tool_results.append(
                {"task_id": task["id"], "tool": tool_name, "success": result.success, "error": result.error}
            )

        return {"tasks": tasks, "tool_results": tool_results, "sources": sources, "errors": errors, "tool_calls_made": calls_made}

    def _node_check_evidence(self, state: AgentState) -> dict:
        iteration = state.get("iteration", 0) + 1
        analysis = state.get("analysis", {})

        # Simple queries never needed evidence; complex ones only get a
        # handful of extra iterations before we force a stop.
        if analysis.get("query_type") == "simple" or iteration >= self.max_iterations:
            return {"iteration": iteration, "evidence_sufficient": True}

        failed_tasks = [t for t in state.get("tasks", []) if t.get("status") == "FAILED"]
        if not failed_tasks:
            return {"iteration": iteration, "evidence_sufficient": True}

        try:
            summary = "\n".join(f"- {t['description']}: {t.get('result') or 'no result'}" for t in state.get("tasks", []))
            check = self.llm.generate_structured(
                verifier_prompts.build_prompt(state["query"], summary),
                schema=_build_sufficiency_schema(),
                system=verifier_prompts.SYSTEM,
            )
            sufficient = bool(check.sufficient)
        except LLMError:
            # If we can't even check, don't loop forever — accept what we have.
            sufficient = True

        return {"iteration": iteration, "evidence_sufficient": sufficient}

    def _node_retry_plan(self, state: AgentState) -> dict:
        """Adds one small follow-up task for whatever failed, instead of
        re-planning everything from scratch."""
        failed = [t for t in state.get("tasks", []) if t.get("status") == "FAILED"]
        tasks = list(state.get("tasks", []))
        for t in failed[:2]:  # keep the retry bounded
            tasks.append(
                {
                    "id": str(uuid.uuid4()),
                    "description": t["description"],
                    "status": "PENDING",
                    "assigned_tool": "none",
                    "result": None,
                }
            )
        return {"tasks": tasks}

    def _node_synthesize_answer(self, state: AgentState) -> dict:
        sources = evidence_mod.process_sources(state.get("sources", []), state["query"])
        evidence_blocks = []
        for t in state.get("tasks", []):
            if t.get("result"):
                evidence_blocks.append(fence_untrusted_content(f"[{t['id'][:8]}] {t['result']}"))
        for s in sources:
            snippet = s.get("snippet") or ""
            evidence_blocks.append(fence_untrusted_content(f"[{s['source_id']}] {s.get('title')}: {snippet}"))

        try:
            answer = self.llm.generate(
                synthesizer_prompts.build_prompt(state["query"], evidence_blocks),
                system=synthesizer_prompts.SYSTEM,
            )
        except LLMError as exc:
            answer = ""
            state.setdefault("errors", []).append(str(exc))

        status = "completed" if answer else "failed"
        return {"final_answer": answer, "sources": sources, "status": status}

    # ---- graph construction ------------------------------------------

    def _build_runner(self) -> Callable[[AgentState], AgentState]:
        if _HAS_LANGGRAPH:
            return self._build_langgraph_runner()
        logger.warning("LangGraph not available; using fallback sequential runner.")
        return self._run_fallback

    def _build_langgraph_runner(self) -> Callable[[AgentState], AgentState]:
        graph = StateGraph(AgentState)
        graph.add_node("analyze_query", self._node_analyze_query)
        graph.add_node("create_plan", self._node_create_plan)
        graph.add_node("execute_tools", self._node_execute_tools)
        graph.add_node("check_evidence", self._node_check_evidence)
        graph.add_node("retry_plan", self._node_retry_plan)
        graph.add_node("synthesize_answer", self._node_synthesize_answer)

        graph.add_edge(START, "analyze_query")
        graph.add_edge("analyze_query", "create_plan")

        def _has_tasks(state: AgentState) -> str:
            return "execute_tools" if state.get("tasks") else "synthesize_answer"

        graph.add_conditional_edges("create_plan", _has_tasks, {
            "execute_tools": "execute_tools",
            "synthesize_answer": "synthesize_answer",
        })
        graph.add_edge("execute_tools", "check_evidence")

        def _sufficient(state: AgentState) -> str:
            return "synthesize_answer" if state.get("evidence_sufficient") else "retry_plan"

        graph.add_conditional_edges("check_evidence", _sufficient, {
            "synthesize_answer": "synthesize_answer",
            "retry_plan": "retry_plan",
        })
        graph.add_edge("retry_plan", "execute_tools")
        graph.add_edge("synthesize_answer", END)

        compiled = graph.compile()

        def run(state: AgentState) -> AgentState:
            return compiled.invoke(state)

        return run

    def _run_fallback(self, state: AgentState) -> AgentState:
        """A plain-Python equivalent of the graph above, used only if
        LangGraph can't be imported."""
        state.update(self._node_analyze_query(state))
        state.update(self._node_create_plan(state))
        if state.get("tasks"):
            while True:
                state.update(self._node_execute_tools(state))
                state.update(self._node_check_evidence(state))
                if state.get("evidence_sufficient"):
                    break
                state.update(self._node_retry_plan(state))
        state.update(self._node_synthesize_answer(state))
        return state

    # ---- public API ------------------------------------------------------

    def run(self, query: str, depth: str = "basic", document_id: str | None = None) -> AgentState:
        state = new_state(query=query, depth=depth, document_id=document_id)
        start = time.time()
        try:
            result = self._runner(state)
        except Exception as exc:  # noqa: BLE001 - never let the graph crash the request
            logger.exception("Agent run failed")
            state["errors"] = state.get("errors", []) + [str(exc)]
            state["status"] = "failed"
            state["final_answer"] = ""
            result = state
        result["_duration_seconds"] = time.time() - start
        return result


def _extract_expression(text: str) -> str:
    """Pull the arithmetic-looking substring out of a task description
    like 'Calculate 25 * 18' so the calculator gets a clean expression."""
    import re

    match = re.search(r"[-+0-9\.\s\*\/\%\(\)\^]{3,}", text)
    return match.group(0).strip() if match else text


def _build_sufficiency_schema():
    from pydantic import BaseModel

    class Sufficiency(BaseModel):
        sufficient: bool
        missing: str = ""

    return Sufficiency
