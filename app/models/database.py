"""SQLAlchemy models and a small init helper.

Kept intentionally simple: one file, five small tables, no migrations
framework (SQLite create_all is enough for a portfolio project).
"""
from __future__ import annotations

import datetime as dt
import uuid

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def _uuid() -> str:
    return str(uuid.uuid4())


class Document(db.Model):
    __tablename__ = "documents"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    filename = db.Column(db.String(255), nullable=False)
    original_filename = db.Column(db.String(255), nullable=False)
    extension = db.Column(db.String(10), nullable=False)
    size_bytes = db.Column(db.Integer, nullable=False)
    chunk_count = db.Column(db.Integer, default=0)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "filename": self.original_filename,
            "extension": self.extension,
            "size_bytes": self.size_bytes,
            "chunk_count": self.chunk_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ResearchSession(db.Model):
    __tablename__ = "research_sessions"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    query = db.Column(db.Text, nullable=False)
    depth = db.Column(db.String(20), default="basic")
    status = db.Column(db.String(20), default="pending")
    answer = db.Column(db.Text, default="")
    duration_seconds = db.Column(db.Float, nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=dt.datetime.utcnow, onupdate=dt.datetime.utcnow)

    tasks = db.relationship("ResearchTask", backref="session", cascade="all, delete-orphan")
    sources = db.relationship("Source", backref="session", cascade="all, delete-orphan")
    tool_executions = db.relationship("ToolExecution", backref="session", cascade="all, delete-orphan")

    def to_dict(self, include_children: bool = True) -> dict:
        data = {
            "session_id": self.id,
            "query": self.query,
            "depth": self.depth,
            "status": self.status,
            "answer": self.answer,
            "duration_seconds": self.duration_seconds,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
        if include_children:
            data["tasks"] = [t.to_dict() for t in self.tasks]
            data["sources"] = [s.to_dict() for s in self.sources]
        return data


class ResearchTask(db.Model):
    __tablename__ = "research_tasks"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    session_id = db.Column(db.String(36), db.ForeignKey("research_sessions.id"))
    description = db.Column(db.Text, nullable=False)
    status = db.Column(db.String(20), default="PENDING")
    assigned_tool = db.Column(db.String(50), nullable=True)
    result = db.Column(db.Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "assigned_tool": self.assigned_tool,
            "result": self.result,
        }


class Source(db.Model):
    __tablename__ = "sources"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    session_id = db.Column(db.String(36), db.ForeignKey("research_sessions.id"))
    title = db.Column(db.String(255), nullable=False)
    url = db.Column(db.String(1024), nullable=True)
    document = db.Column(db.String(255), nullable=True)
    publisher = db.Column(db.String(255), nullable=True)
    snippet = db.Column(db.Text, nullable=True)

    def to_dict(self) -> dict:
        return {
            "source_id": self.id,
            "title": self.title,
            "url": self.url,
            "document": self.document,
            "publisher": self.publisher,
            "snippet": self.snippet,
        }


class ToolExecution(db.Model):
    __tablename__ = "tool_executions"

    id = db.Column(db.String(36), primary_key=True, default=_uuid)
    session_id = db.Column(db.String(36), db.ForeignKey("research_sessions.id"))
    tool_name = db.Column(db.String(50), nullable=False)
    success = db.Column(db.Boolean, default=False)
    duration_seconds = db.Column(db.Float, nullable=True)
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=dt.datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "success": self.success,
            "duration_seconds": self.duration_seconds,
            "error": self.error,
        }


def init_db(app) -> None:
    db.init_app(app)
    with app.app_context():
        db.create_all()
