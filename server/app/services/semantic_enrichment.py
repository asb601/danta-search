"""Second-pass semantic enrichment — workflow-aware good_for generation.

Runs AFTER semantic roles, relationships, and semantic layer are built.
Enriches good_for with operational workflow semantics derived from:
  1. Same-role-kind column groups (ratio/completion/reconciliation signals)
  2. Approved SemanticRelationship neighbors (cross-table workflow signals)

Bounded, fail-open, idempotent, domain-free.
"""
from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import ingest_logger
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticEntity, SemanticRelationship
from app.retrieval.embeddings import build_search_text, embed_text
from app.services.semantic_roles import dynamic_role_label, role_kind

_MAX_NEIGHBORS = 3
_MAX_ROLE_GROUPS = 5
_MAX_COLS_PER_GROUP = 5


def _group_same_role_kind(column_semantic_roles: dict[str, str]) -> list[dict[str, Any]]:
    """Group columns by role kind. Only multi-column groups (2+ cols) are returned.

    Returns list of {kind, label, columns, count} ordered by group size descending.
    """
    by_kind: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for col, role in column_semantic_roles.items():
        kind = role_kind(role)
        if kind is None:
            continue
        label = dynamic_role_label(role) or role
        by_kind[kind][label].append(col)

    groups: list[dict[str, Any]] = []
    for kind, label_map in by_kind.items():
        for label, cols in label_map.items():
            if len(cols) >= 2:
                groups.append({
                    "kind": kind,
                    "label": label,
                    "columns": cols[:_MAX_COLS_PER_GROUP],
                    "count": len(cols),
                })
    groups.sort(key=lambda g: g["count"], reverse=True)
    return groups[:_MAX_ROLE_GROUPS]


async def _load_approved_neighbors(
    file_id: str, db: AsyncSession
) -> list[dict[str, Any]]:
    """Return approved SemanticRelationship neighbors with their descriptions.

    Ordered by confidence_score DESC, capped at _MAX_NEIGHBORS.
    """
    rows = (
        await db.execute(
            select(SemanticRelationship)
            .where(
                (
                    (SemanticRelationship.file_a_id == file_id)
                    | (SemanticRelationship.file_b_id == file_id)
                ),
                SemanticRelationship.approval_status == "approved",
                SemanticRelationship.status == "active",
            )
            .order_by(SemanticRelationship.confidence_score.desc())
            .limit(_MAX_NEIGHBORS)
        )
    ).scalars().all()

    neighbors: list[dict[str, Any]] = []
    for rel in rows:
        neighbor_file_id = (
            rel.file_b_id if rel.file_a_id == file_id else rel.file_a_id
        )
        neighbor_meta = (
            await db.execute(
                select(FileMetadata).where(FileMetadata.file_id == neighbor_file_id)
            )
        ).scalar_one_or_none()
        if not neighbor_meta:
            continue
        neighbor_file = await db.get(File, neighbor_file_id)
        join_col_this = rel.from_column if rel.file_a_id == file_id else rel.to_column
        join_col_neighbor = (
            rel.to_column if rel.file_a_id == file_id else rel.from_column
        )
        neighbors.append({
            "name": (
                neighbor_file.name
                if neighbor_file
                else (neighbor_meta.blob_path or neighbor_file_id)
            ),
            "relationship_type": rel.relationship_type,
            "join_column_this": join_col_this,
            "join_column_neighbor": join_col_neighbor,
            "neighbor_description": (neighbor_meta.ai_description or "")[:300],
            "neighbor_good_for": (neighbor_meta.good_for or [])[:4],
            "confidence_score": float(rel.confidence_score or 0.0),
        })
    return neighbors


def _dedup_additions(existing: list[str], additions: list[str]) -> list[str]:
    """Remove additions too similar to existing phrases (token overlap > 70%)."""
    existing_lower = {p.lower().strip() for p in existing}
    deduped: list[str] = []
    for addition in additions:
        low = addition.lower().strip()
        if low in existing_lower:
            continue
        tokens = set(low.split())
        if any(
            len(tokens & set(ex.split())) / max(len(tokens), 1) > 0.7
            for ex in existing_lower
        ):
            continue
        deduped.append(addition)
        existing_lower.add(low)
    return deduped


async def run_semantic_enrichment_for_file(
    file_id: str, db: AsyncSession
) -> dict[str, Any]:
    """Enrich good_for with workflow-aware phrases; rebuild search_text + embedding.

    Returns:
        dict with keys: skipped (bool), reason (str, optional),
        additions (int), neighbors_used (int), role_groups_used (int), duration_ms (float)
    """
    start = time.perf_counter()
    metadata = (
        await db.execute(
            select(FileMetadata).where(FileMetadata.file_id == file_id)
        )
    ).scalar_one_or_none()
    if not metadata:
        return {"skipped": True, "reason": "no_metadata", "additions": 0, "duration_ms": 0.0}

    file = await db.get(File, file_id)
    filename = file.name if file else (metadata.blob_path or file_id)
    roles: dict[str, str] = metadata.column_semantic_roles or {}
    existing_good_for: list[str] = list(metadata.good_for or [])
    current_description: str = metadata.ai_description or ""

    entity = (
        await db.execute(
            select(SemanticEntity).where(SemanticEntity.file_id == file_id)
        )
    ).scalar_one_or_none()
    grain: str | None = entity.grain if entity else None

    role_groups = _group_same_role_kind(roles)
    neighbors = await _load_approved_neighbors(file_id, db)

    if not role_groups and not neighbors:
        return {
            "skipped": True,
            "reason": "no_enrichment_context",
            "additions": 0,
            "duration_ms": round((time.perf_counter() - start) * 1000, 2),
        }

    from app.core.llm_tasks import enrich_semantic_description  # noqa: PLC0415

    result = await enrich_semantic_description(
        filename=filename,
        current_description=current_description,
        current_good_for=existing_good_for,
        role_groups=role_groups,
        neighbors=neighbors,
        grain=grain,
    )

    raw_additions: list[str] = result.get("additional_good_for", [])
    if not isinstance(raw_additions, list):
        raw_additions = []
    raw_additions = [str(s).strip() for s in raw_additions if s and str(s).strip()]
    new_additions = _dedup_additions(existing_good_for, raw_additions)

    if new_additions:
        metadata.good_for = existing_good_for + new_additions
        metadata.search_text = build_search_text(metadata)
        metadata.description_embedding = await embed_text(metadata.search_text)
        await db.commit()

    duration = round((time.perf_counter() - start) * 1000, 2)
    ingest_logger.info(
        "semantic_enrichment",
        file_id=file_id,
        filename=filename,
        additions=len(new_additions),
        neighbors_used=len(neighbors),
        role_groups_used=len(role_groups),
        duration_ms=duration,
    )
    return {
        "skipped": False,
        "additions": len(new_additions),
        "neighbors_used": len(neighbors),
        "role_groups_used": len(role_groups),
        "duration_ms": duration,
    }
