"""Discovery evidence records for request-time source selection.

This module scores metadata evidence only. It does not authorize SQL, infer
joins, or require relationship edges. Relationship data can be added as one
more evidence signal later, but discovery must not depend on it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.embeddings import build_search_text

_SHORT_FUNCTION_WORDS = {"of", "in", "on", "by", "to", "an", "or", "as", "at", "is", "be", "me", "my", "we", "us", "all", "any", "each"}
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[&'][a-z0-9]+)*", re.I)
_QUERY_INTENT_WORDS = {
    "show", "give", "find", "list", "tell", "fetch", "display",
    "what", "when", "where", "which", "with", "from", "have",
    "does", "that", "this", "them", "they", "will", "been",
    "were", "much", "many", "also", "just", "more", "than",
}


def _tokenize(text: str, *, min_length: int = 2) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(str(text or "").casefold()):
        normalized = re.sub(r"[^a-z0-9]+", "", match)
        is_compound = len(normalized) < len(match)
        if not is_compound and len(normalized) < min_length:
            continue
        if normalized in _QUERY_INTENT_WORDS:
            continue
        tokens.append(normalized)
    return tokens


def _search_tokens(text: str) -> list[str]:
    return [
        token for token in _tokenize(str(text or ""), min_length=2)
        if token not in _SHORT_FUNCTION_WORDS
    ]


@dataclass(frozen=True)
class DiscoveryCandidateEvidence:
    file_id: str
    blob_path: str
    score: float
    retrieval_score: float = 0.0
    channels: list[str] = field(default_factory=list)
    matched_terms: list[str] = field(default_factory=list)
    matched_columns: list[str] = field(default_factory=list)
    matched_search_queries: list[str] = field(default_factory=list)
    evidence_reasons: list[str] = field(default_factory=list)
    source_anchor_match_count: int = 0
    output_match_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "file_id": self.file_id,
            "blob_path": self.blob_path,
            "score": round(float(self.score), 4),
            "retrieval_score": round(float(self.retrieval_score), 4),
            "channels": self.channels,
            "matched_terms": self.matched_terms,
            "matched_columns": self.matched_columns,
            "matched_search_queries": self.matched_search_queries,
            "evidence_reasons": self.evidence_reasons,
            "source_anchor_match_count": self.source_anchor_match_count,
            "output_match_count": self.output_match_count,
        }


def _dedup(items: list[str], *, limit: int = 20) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        value = str(item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
        if len(out) >= limit:
            break
    return out


def _tokens(items: list[str]) -> set[str]:
    tokens: set[str] = set()
    for item in items:
        tokens.update(_search_tokens(str(item or "")))
    return tokens


def _column_names(entry: dict) -> list[str]:
    cols = [
        str(c.get("name"))
        for c in (entry.get("columns_info") or [])
        if isinstance(c, dict) and c.get("name")
    ]
    if not cols:
        cols = [str(c) for c in (entry.get("column_names") or []) if isinstance(c, str)]
    return cols


def _matched_tokens(tokens: set[str], text: str) -> list[str]:
    if not tokens or not text:
        return []
    text_tokens = set(_search_tokens(text))
    return sorted(tokens & text_tokens)


def build_discovery_candidate_evidence(
    *,
    work_order: Any,
    catalog: list[dict],
    retrieved_with_scores: list[tuple[Any, float]] | None = None,
    retrieval_channels: dict[str, list[str]] | None = None,
    retrieval_variant_evidence: dict[str, dict] | None = None,
) -> list[DiscoveryCandidateEvidence]:
    """Return sorted evidence records for candidate source files."""
    score_by_file_id = {
        getattr(meta, "file_id", ""): float(score)
        for meta, score in (retrieved_with_scores or [])
        if getattr(meta, "file_id", None)
    }
    source_tokens = _tokens(list(getattr(work_order, "source_anchor_terms", []) or []))
    output_tokens = _tokens(list(getattr(work_order, "requested_outputs", []) or []))
    evidence_tokens = _tokens(list(getattr(work_order, "source_evidence_needs", []) or []))
    search_queries = list(getattr(work_order, "candidate_search_queries", []) or [])
    records: list[DiscoveryCandidateEvidence] = []

    for entry in catalog:
        file_id = str(entry.get("file_id") or "")
        blob_path = str(entry.get("blob_path") or "")
        columns = _column_names(entry)
        column_text = " ".join(columns)
        text = " ".join([blob_path, column_text, build_search_text(entry)])

        source_matches = _matched_tokens(source_tokens, text)
        output_matches = _matched_tokens(output_tokens, text)
        evidence_matches = _matched_tokens(evidence_tokens, text)
        matched_columns = [
            col for col in columns
            if _matched_tokens(source_tokens | output_tokens | evidence_tokens, col)
        ][:12]

        matched_queries: list[str] = []
        for variant in search_queries:
            variant_tokens = set(_search_tokens(variant))
            if variant_tokens and len(variant_tokens & set(_search_tokens(text))) >= max(1, min(2, len(variant_tokens))):
                matched_queries.append(variant)

        retrieval_score = float(score_by_file_id.get(file_id, 0.0))
        channels = list((retrieval_channels or {}).get(file_id, []))
        variant_info = (retrieval_variant_evidence or {}).get(file_id, {})
        if variant_info:
            matched_queries.extend(list(variant_info.get("matched_queries") or []))
            channels = _dedup(channels + list(variant_info.get("channels") or []), limit=8)

        reasons: list[str] = []
        if source_matches:
            reasons.append("source_anchor_metadata_match")
        if output_matches:
            reasons.append("requested_output_metadata_match")
        if matched_columns:
            reasons.append("column_evidence_match")
        if channels:
            reasons.append("retrieval_channel_match")
        if matched_queries:
            reasons.append("work_order_search_variant_match")

        score = (
            len(source_matches) * 3.0
            + len(output_matches) * 1.6
            + len(evidence_matches) * 1.2
            + len(matched_columns) * 0.8
            + len(channels) * 0.6
            + min(retrieval_score * 10.0, 2.0)
        )
        if score <= 0.0:
            continue

        records.append(DiscoveryCandidateEvidence(
            file_id=file_id,
            blob_path=blob_path,
            score=round(score, 4),
            retrieval_score=retrieval_score,
            channels=channels,
            matched_terms=_dedup(source_matches + output_matches + evidence_matches, limit=16),
            matched_columns=_dedup(matched_columns, limit=12),
            matched_search_queries=_dedup(matched_queries, limit=6),
            evidence_reasons=_dedup(reasons, limit=8),
            source_anchor_match_count=len(source_matches),
            output_match_count=len(output_matches),
        ))

    records.sort(key=lambda item: (item.score, item.source_anchor_match_count, item.output_match_count), reverse=True)
    return records