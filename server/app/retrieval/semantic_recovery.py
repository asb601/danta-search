"""Semantic Recovery Retriever — staged fallback for failed standard retrieval.

PURPOSE:
  When standard retrieval (BM25 + fuzzy + vector + RRF fusion) returns empty,
  this module attempts structured semantic recovery BEFORE falling back to crude
  keyword scoring over the entire catalog.

RECOVERY STAGES (ordered by quality, aggregated):
  1. Role-cluster matching
     — Find files sharing semantic role labels with resolved query entities.
     — Uses entity_resolution results (already computed, no new DB query).
     — Grounding: "role_cluster"

  2. Graph topology spreading (2-hop)
     — Expand 2 hops from resolver-pinned anchor file_ids via approved edges.
     — At most 2 DB queries (hop-1, hop-2).
     — Grounding: "graph_topology"

  3. Cross-domain semantic bridging
     — Files whose semantic role labels share ANY token with the query entities.
     — Broader than stage 1 (any token vs. entity name overlap).
     — Zero DB queries.
     — Grounding: "semantic_bridge"

  4. Keyword degraded (original behavior, last resort)
     — Score full catalog by keyword overlap with query.
     — Grounding: "keyword_degraded"

RETURN TYPE:
  (candidates: list[dict], grounding_quality: str)
  candidates are lean catalog entries — same shape as shortlist entries.

DESIGN CONSTRAINTS:
  - At most 2 DB queries total (both in stage 2).
  - Stages 1, 3, 4 are zero-query (pure in-memory).
  - Non-fatal: any stage failure falls through to next stage.
    - Stages are merged, deduped, weighted, and ranked; no blind union.
  - column_semantic_roles must be present in catalog entries for stages 1+3.
    Falls back gracefully if absent.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import pipeline_logger

if TYPE_CHECKING:
    pass

_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")

# Minimum entity resolution confidence to use file as a role-cluster anchor
_ROLE_CLUSTER_CONF_FLOOR: float = 0.50

_STAGE_WEIGHTS: dict[str, float] = {
    "role_cluster": 1.00,
    "graph_topology": 0.95,
    "semantic_bridge": 0.80,
    "keyword_degraded": 0.30,
}

_SEMANTIC_RETRIEVAL_CHANNELS: frozenset[str] = frozenset({"vector", "opensearch"})


# ── Public API ─────────────────────────────────────────────────────────────────

async def semantic_recovery_retrieve(
    query: str,
    full_catalog: list[dict],
    entity_resolution: dict,               # entity_name → list[EntityCandidate]
    resolver_pinned_file_ids: list[str],   # high-confidence anchor file_ids
    db: AsyncSession,
    top_k: int,
    container_id: str | None,
    retrieval_channels: dict[str, list[str]] | None = None,
) -> tuple[list[dict], str]:
    """
    Attempt staged semantic recovery when standard retrieval returns empty.

    Returns:
        (ranked_candidates, grounding_quality)
    """
    q_words = _tokenize(query)
    full_by_id: dict[str, dict] = {
        e["file_id"]: e for e in full_catalog if e.get("file_id")
    }
    retrieval_channels = retrieval_channels or {}
    merged: dict[str, dict] = {}

    def add_stage(stage: str, entries: list[dict]) -> None:
        ranked = _rank_entries_with_scores(entries, q_words)
        if not ranked:
            return
        max_kw = max((score for _, score in ranked), default=0.0) or 1.0
        for rank, (entry, kw_score) in enumerate(ranked):
            fid = entry.get("file_id")
            if not fid:
                continue
            hit = merged.setdefault(fid, {
                "entry": entry,
                "score": 0.0,
                "stages": set(),
            })
            stage_weight = _STAGE_WEIGHTS.get(stage, 0.20)
            keyword_component = min(kw_score / max_kw, 1.0) * 0.20 if kw_score > 0 else 0.0
            rank_component = 0.08 / max(rank + 1, 1)
            semantic_component = _retrieval_semantic_bonus(fid, retrieval_channels)
            hit["score"] += stage_weight + keyword_component + rank_component + semantic_component
            hit["stages"].add(stage)

    # ── Stage 1: Role-cluster matching ────────────────────────────────────────
    if entity_resolution:
        role_matched = _match_by_role_clusters(entity_resolution, full_catalog)
        if role_matched:
            add_stage("role_cluster", [full_by_id[fid] for fid in role_matched if fid in full_by_id])

    # ── Stage 2: Graph topology spreading (2-hop) ─────────────────────────────
    if resolver_pinned_file_ids:
        try:
            two_hop = await _two_hop_expansion(resolver_pinned_file_ids, db)
            new_ids = [fid for fid in two_hop if fid not in set(resolver_pinned_file_ids)]
            if new_ids:
                add_stage("graph_topology", [full_by_id[fid] for fid in new_ids if fid in full_by_id])
        except Exception as exc:
            pipeline_logger.warning("semantic_recovery_graph_error", error=str(exc)[:200])

    # ── Stage 3: Cross-domain semantic bridging ───────────────────────────────
    bridge_ids = _bridge_by_shared_roles(entity_resolution, full_catalog)
    if bridge_ids:
        add_stage("semantic_bridge", [full_by_id[fid] for fid in bridge_ids if fid in full_by_id])

    # ── Stage 4: Keyword degraded evidence ───────────────────────────────────
    # Always contributes bounded low-weight evidence. It cannot override files
    # supported by semantic or graph stages, but it gives the planner a fallback
    # when all structural stages are empty.
    add_stage("keyword_degraded", _rank_by_keywords(full_catalog, q_words)[:top_k])

    if not merged:
        return [], "keyword_degraded"

    ranked_hits = sorted(
        merged.values(),
        key=lambda h: (h["score"], len(h["stages"])),
        reverse=True,
    )
    candidates = [h["entry"] for h in ranked_hits[:top_k]]
    stage_names = sorted({stage for h in ranked_hits for stage in h["stages"]})
    grounding_quality = "aggregated:" + "+".join(stage_names)

    pipeline_logger.info(
        "semantic_recovery",
        stage="aggregated",
        stages=stage_names,
        matched=len(candidates),
        total_catalog=len(full_catalog),
        query=query[:80],
    )
    return candidates, grounding_quality


# ── Stage 1: role-cluster matching ────────────────────────────────────────────

def _match_by_role_clusters(
    entity_resolution: dict,
    full_catalog: list[dict],
) -> list[str]:
    """
    Find files sharing semantic role labels with any resolved query entity.
    Uses high-confidence candidates from entity_resolution as anchors.
    """
    target_labels: set[str] = set()
    for entity_name, candidates in entity_resolution.items():
        for c in (candidates or []):
            conf = getattr(c, "confidence", 0.0)
            if conf >= _ROLE_CLUSTER_CONF_FLOOR:
                target_labels.add(entity_name.lower())

    if not target_labels:
        return []

    matched: list[str] = []
    for entry in full_catalog:
        fid = entry.get("file_id")
        if not fid:
            continue
        roles: dict = entry.get("column_semantic_roles") or {}
        for role_val in roles.values():
            parsed = _parse_role(str(role_val) if role_val else None)
            if not parsed:
                continue
            _, label = parsed
            label_toks = frozenset(
                t for t in re.split(r"[^a-z0-9]+", label.lower()) if t
            )
            for tgt in target_labels:
                tgt_toks = frozenset(
                    t for t in re.split(r"[^a-z0-9]+", tgt) if t
                )
                if label_toks & tgt_toks:
                    if fid not in matched:
                        matched.append(fid)
                    break
    return matched


# ── Stage 2: 2-hop graph expansion ────────────────────────────────────────────

async def _two_hop_expansion(
    anchor_ids: list[str],
    db: AsyncSession,
) -> list[str]:
    """Expand 2 hops from anchor file_ids via approved SemanticRelationship edges."""
    from app.models.semantic_layer import SemanticRelationship

    # Hop 1
    stmt1 = (
        select(SemanticRelationship.file_a_id, SemanticRelationship.file_b_id)
        .where(
            or_(
                SemanticRelationship.file_a_id.in_(anchor_ids),
                SemanticRelationship.file_b_id.in_(anchor_ids),
            ),
            SemanticRelationship.status == "active",
            SemanticRelationship.approval_status == "approved",
        )
    )
    rows1 = (await db.execute(stmt1)).all()

    hop1_ids: set[str] = set(anchor_ids)
    for row in rows1:
        hop1_ids.add(row.file_a_id)
        hop1_ids.add(row.file_b_id)

    new_hop1 = hop1_ids - set(anchor_ids)
    if not new_hop1:
        return list(hop1_ids)

    # Hop 2
    stmt2 = (
        select(SemanticRelationship.file_a_id, SemanticRelationship.file_b_id)
        .where(
            or_(
                SemanticRelationship.file_a_id.in_(list(new_hop1)),
                SemanticRelationship.file_b_id.in_(list(new_hop1)),
            ),
            SemanticRelationship.status == "active",
            SemanticRelationship.approval_status == "approved",
        )
    )
    rows2 = (await db.execute(stmt2)).all()

    hop2_ids: set[str] = set(hop1_ids)
    for row in rows2:
        hop2_ids.add(row.file_a_id)
        hop2_ids.add(row.file_b_id)

    return list(hop2_ids)


# ── Stage 3: semantic bridging ────────────────────────────────────────────────

def _bridge_by_shared_roles(
    entity_resolution: dict,
    full_catalog: list[dict],
) -> list[str]:
    """
    Broader search: files whose semantic role labels share ANY meaningful token
    with any query entity name. Lower precision than stage 1, higher recall.
    """
    if not entity_resolution:
        return []

    entity_tokens: set[str] = set()
    for entity_name in entity_resolution:
        entity_tokens.update(
            t for t in re.split(r"[^a-z0-9]+", entity_name.lower())
            if len(t) >= 3
        )

    if not entity_tokens:
        return []

    matched: list[str] = []
    for entry in full_catalog:
        fid = entry.get("file_id")
        if not fid:
            continue
        roles: dict = entry.get("column_semantic_roles") or {}
        for role_val in roles.values():
            parsed = _parse_role(str(role_val) if role_val else None)
            if not parsed:
                continue
            _, label = parsed
            label_toks = frozenset(
                t for t in re.split(r"[^a-z0-9]+", label.lower()) if len(t) >= 3
            )
            if label_toks & entity_tokens:
                if fid not in matched:
                    matched.append(fid)
                break

    return matched


# ── Shared helpers ─────────────────────────────────────────────────────────────

def _parse_role(role_str: str | None) -> tuple[str, str] | None:
    if not role_str:
        return None
    m = _ROLE_RE.match(str(role_str))
    return (m.group(1), m.group(2)) if m else None


def _tokenize(query: str) -> list[str]:
    """Extract meaningful tokens from query for keyword scoring."""
    _STOP = frozenset({
        "the", "and", "for", "with", "from", "that", "this", "show", "give",
        "list", "get", "all", "what", "where", "how", "when", "who", "which",
        "are", "was", "were", "have", "has", "had", "can", "will", "would",
    })
    tokens = re.split(r"[^a-z0-9]+", query.lower())
    return [t for t in tokens if len(t) >= 4 and t not in _STOP]


def _kw_score(entry: dict, q_words: list[str]) -> float:
    blob = (entry.get("blob_path") or "").lower()
    desc = (entry.get("ai_description") or "").lower()
    col_text = " ".join(
        c.get("name", "") if isinstance(c, dict) else str(c)
        for c in (entry.get("column_names") or [])
    ).lower()
    score = 0.0
    for w in q_words:
        if w in blob:
            score += 3.0
        if w in col_text:
            score += 2.0
        if w in desc:
            score += 1.0
    # Boost lookup/dimension tables to surface master-data
    if entry.get("good_for") and any(
        "lookup" in str(g).lower() or "master" in str(g).lower() or "dimension" in str(g).lower()
        for g in entry["good_for"]
    ):
        score += 0.5
    return score


def _rank_by_keywords(entries: list[dict], q_words: list[str]) -> list[dict]:
    return sorted(entries, key=lambda e: _kw_score(e, q_words), reverse=True)


def _rank_entries_with_scores(entries: list[dict], q_words: list[str]) -> list[tuple[dict, float]]:
    ranked = [(entry, _kw_score(entry, q_words)) for entry in entries]
    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def _retrieval_semantic_bonus(fid: str, retrieval_channels: dict[str, list[str]]) -> float:
    channels = set(retrieval_channels.get(fid, []) or [])
    if channels & _SEMANTIC_RETRIEVAL_CHANNELS:
        return 0.15
    return 0.0
