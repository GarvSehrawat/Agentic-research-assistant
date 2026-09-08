"""Lightweight evidence processing: dedupe near-identical snippets, rank
by simple relevance, and cap total context size before synthesis.

Deliberately simple — no embedding-based reranking here, just something
that keeps the final prompt small and non-redundant.
"""
from __future__ import annotations

from app.agent.state import SourceDict

MAX_EVIDENCE_CHARS = 6000


def process_sources(sources: list[SourceDict], query: str) -> list[SourceDict]:
    deduped = _dedupe(sources)
    ranked = _rank(deduped, query)
    return _cap(ranked, MAX_EVIDENCE_CHARS)


def _dedupe(sources: list[SourceDict]) -> list[SourceDict]:
    seen: set[str] = set()
    result: list[SourceDict] = []
    for s in sources:
        key = (s.get("snippet") or "")[:120].strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        result.append(s)
    return result


def _rank(sources: list[SourceDict], query: str) -> list[SourceDict]:
    query_terms = {w.lower() for w in query.split() if len(w) > 3}

    def score(s: SourceDict) -> int:
        text = (s.get("snippet") or "").lower()
        return sum(1 for term in query_terms if term in text)

    return sorted(sources, key=score, reverse=True)


def _cap(sources: list[SourceDict], max_chars: int) -> list[SourceDict]:
    total = 0
    result: list[SourceDict] = []
    for s in sources:
        length = len(s.get("snippet") or "")
        if total + length > max_chars and result:
            break
        total += length
        result.append(s)
    return result
