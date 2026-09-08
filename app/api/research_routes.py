"""Research endpoints: kick off an agent run, inspect a session, cancel one."""
from __future__ import annotations

import logging
import time

from flask import Blueprint, current_app, jsonify, request
from pydantic import ValidationError

from app.agent.graph import ResearchAgent
from app.models.database import ResearchSession, ResearchTask, Source, ToolExecution, db
from app.schemas.research import ResearchRequest

logger = logging.getLogger(__name__)
research_bp = Blueprint("research", __name__)


def _error(code: str, message: str, status: int):
    return jsonify({"error": {"code": code, "message": message}}), status


@research_bp.post("/research")
def run_research():
    payload = request.get_json(silent=True)
    if payload is None:
        return _error("INVALID_REQUEST", "Request body must be valid JSON.", 400)

    try:
        req = ResearchRequest.model_validate(payload)
    except ValidationError as exc:
        return _error("INVALID_REQUEST", str(exc.errors()), 400)

    llm = current_app.extensions.get("llm_provider")
    if llm is None:
        return _error(
            "LLM_UNAVAILABLE",
            "No LLM provider is configured. Set OPENAI_API_KEY in your .env.",
            503,
        )

    tools = current_app.extensions["tool_registry"]
    cfg = current_app.config
    agent = ResearchAgent(
        llm=llm,
        tools=tools,
        max_iterations=cfg["MAX_AGENT_ITERATIONS"],
        max_tool_calls=cfg["MAX_TOOL_CALLS"],
        max_tasks=cfg["MAX_TASKS"],
    )

    session = ResearchSession(query=req.query, depth=req.depth.value, status="running")
    db.session.add(session)
    db.session.commit()

    start = time.time()
    try:
        result_state = agent.run(query=req.query, depth=req.depth.value)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Agent run raised unexpectedly")
        session.status = "failed"
        session.answer = ""
        db.session.commit()
        return _error("AGENT_FAILURE", f"Research failed: {exc}", 500)

    duration = time.time() - start
    session.status = result_state.get("status", "completed")
    session.answer = result_state.get("final_answer", "")
    session.duration_seconds = duration

    for task in result_state.get("tasks", []):
        db.session.add(
            ResearchTask(
                session_id=session.id,
                description=task.get("description", ""),
                status=task.get("status", "PENDING"),
                assigned_tool=task.get("assigned_tool"),
                result=task.get("result"),
            )
        )
    for src in result_state.get("sources", []):
        db.session.add(
            Source(
                session_id=session.id,
                title=src.get("title", "Untitled"),
                url=src.get("url"),
                document=src.get("document"),
                publisher=src.get("publisher"),
                snippet=src.get("snippet"),
            )
        )
    for tr in result_state.get("tool_results", []):
        db.session.add(
            ToolExecution(
                session_id=session.id,
                tool_name=tr.get("tool", "unknown"),
                success=bool(tr.get("success")),
                error=tr.get("error"),
            )
        )
    db.session.commit()

    return jsonify(session.to_dict()), 200


@research_bp.get("/research/<session_id>")
def get_research(session_id: str):
    session = db.session.get(ResearchSession, session_id)
    if session is None:
        return _error("NOT_FOUND", "No research session with that id.", 404)
    return jsonify(session.to_dict()), 200


@research_bp.post("/research/<session_id>/cancel")
def cancel_research(session_id: str):
    session = db.session.get(ResearchSession, session_id)
    if session is None:
        return _error("NOT_FOUND", "No research session with that id.", 404)
    if session.status in ("completed", "failed", "cancelled"):
        return _error("INVALID_STATE", f"Session already {session.status}; cannot cancel.", 409)
    session.status = "cancelled"
    db.session.commit()
    return jsonify(session.to_dict()), 200
