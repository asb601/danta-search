"""Normalization helpers for governed semantic memory.

The memory store is intentionally role-kind driven. Business labels come from
ingestion evidence; this module only normalizes shape, status, and terms.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


MEMORY_TYPES = frozenset({
    "entity",
    "metric",
    "dimension",
    "temporal",
    "relationship",
    "capability",
    "grain",
    "null_semantic",
})
GOVERNANCE_STATUSES = frozenset({"candidate", "active", "rejected", "deprecated"})
RECORD_STATUSES = frozenset({"active", "inactive", "deprecated"})

_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_]{1,80}", re.IGNORECASE)
_SLUG_RE = re.compile(r"[^a-z0-9]+")
_MAX_TERMS = 40
_MAX_SUMMARY_CHARS = 500


@dataclass(frozen=True)
class MemoryEvidenceInput:
    source_type: str
    source_id: str | None
    evidence_key: str
    evidence_value: dict | list | str | None
    confidence_score: float = 0.0
    file_id: str | None = None


@dataclass(frozen=True)
class MemoryCandidate:
    container_id: str
    memory_type: str
    canonical_key: str
    title: str
    summary: str | None = None
    normalized_terms: list[str] = field(default_factory=list)
    behaviors: list[str] = field(default_factory=list)
    dimensions: dict[str, Any] = field(default_factory=dict)
    constraints: dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.0
    authority_score: float = 0.0
    governance_status: str = "candidate"
    status: str = "active"
    source: str = "ingestion"
    source_file_id: str | None = None
    source_entity_id: str | None = None
    source_relationship_id: str | None = None
    evidence: list[MemoryEvidenceInput] = field(default_factory=list)


def slugify(value: str | None, *, fallback: str = "item") -> str:
    slug = _SLUG_RE.sub("_", (value or "").strip().lower()).strip("_")
    return (slug or fallback)[:120]


def split_role(role: str | None) -> tuple[str | None, str | None]:
    if not role or not role.startswith("custom:"):
        return None, None
    parts = role.split(":", 2)
    if len(parts) != 3:
        return None, None
    return parts[1], parts[2]


def terms_from_text(*values: Any) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            raw = " ".join(str(v) for v in value if v is not None)
        elif isinstance(value, dict):
            raw = " ".join(f"{k} {v}" for k, v in value.items())
        else:
            raw = str(value)
        for token in _TOKEN_RE.findall(raw.lower().replace("-", "_")):
            token = slugify(token)
            if len(token) < 2 or token in seen:
                continue
            terms.append(token)
            seen.add(token)
            if len(terms) >= _MAX_TERMS:
                return terms
    return terms


def canonical_key(memory_type: str, *parts: Any) -> str:
    cleaned = [slugify(str(part)) for part in parts if part is not None and str(part).strip()]
    raw = ":".join([memory_type, *cleaned]) or memory_type
    if len(raw) <= 240:
        return raw
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]
    return f"{raw[:210]}:{digest}"


def bounded_score(value: float | int | None) -> float:
    try:
        return round(max(0.0, min(1.0, float(value or 0.0))), 4)
    except (TypeError, ValueError):
        return 0.0


def normalise_candidate(candidate: MemoryCandidate) -> MemoryCandidate | None:
    if candidate.memory_type not in MEMORY_TYPES:
        return None
    if not candidate.container_id or not candidate.canonical_key or not candidate.title:
        return None

    terms = terms_from_text(
        candidate.normalized_terms,
        candidate.title,
        candidate.summary,
        candidate.behaviors,
        candidate.dimensions,
    )
    governance_status = (
        candidate.governance_status
        if candidate.governance_status in GOVERNANCE_STATUSES
        else "candidate"
    )
    status = candidate.status if candidate.status in RECORD_STATUSES else "active"
    summary = (candidate.summary or "").strip() or None
    if summary and len(summary) > _MAX_SUMMARY_CHARS:
        summary = summary[:_MAX_SUMMARY_CHARS].rsplit(" ", 1)[0] + "..."

    return MemoryCandidate(
        container_id=candidate.container_id,
        memory_type=candidate.memory_type,
        canonical_key=candidate.canonical_key[:255],
        title=candidate.title.strip()[:255],
        summary=summary,
        normalized_terms=terms,
        behaviors=terms_from_text(candidate.behaviors)[:20],
        dimensions=dict(candidate.dimensions or {}),
        constraints=dict(candidate.constraints or {}),
        confidence_score=bounded_score(candidate.confidence_score),
        authority_score=bounded_score(candidate.authority_score),
        governance_status=governance_status,
        status=status,
        source=candidate.source[:50] or "ingestion",
        source_file_id=candidate.source_file_id,
        source_entity_id=candidate.source_entity_id,
        source_relationship_id=candidate.source_relationship_id,
        evidence=list(candidate.evidence or []),
    )