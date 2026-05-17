"""Retrieval graph expansion through approved semantic relationships."""
from __future__ import annotations

from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticRelationship
from app.retrieval.filters import build_base_query
from app.services.semantic_policy import get_semantic_policy


async def graph_expand(
    seed_file_ids: list[str],
    user_id: str,
    is_admin: bool,
    db: AsyncSession,
    min_confidence: float | None = None,
    limit: int = 20,
    allowed_domains: list[str] | None = None,
    container_id: str | None = None,
) -> list[tuple[FileMetadata, float]]:
    """Expand seed files through approved semantic relationships only.

    Raw technical relationships are candidates. They do not enter graph
    expansion until the semantic layer approves them.
    """
    if not seed_file_ids:
        return []

    policy = get_semantic_policy()
    min_confidence = (
        policy.graph_expand_min_confidence
        if min_confidence is None
        else min_confidence
    )
    seed_set = set(seed_file_ids)

    edge_q = (
        select(
            SemanticRelationship.file_a_id,
            SemanticRelationship.file_b_id,
            SemanticRelationship.confidence_score,
        )
        .where(
            or_(
                SemanticRelationship.file_a_id.in_(seed_file_ids),
                SemanticRelationship.file_b_id.in_(seed_file_ids),
            ),
            SemanticRelationship.status == "active",
            SemanticRelationship.approval_status == "approved",
            SemanticRelationship.confidence_score >= min_confidence,
        )
        .order_by(SemanticRelationship.confidence_score.desc())
    )
    if container_id:
        edge_q = edge_q.where(SemanticRelationship.container_id == container_id)

    edge_rows = (await db.execute(edge_q)).all()
    if not edge_rows:
        return []

    neighbour_score: dict[str, float] = {}
    for file_a_id, file_b_id, confidence in edge_rows:
        neighbour_id = file_b_id if file_a_id in seed_set else file_a_id
        if neighbour_id in seed_set:
            continue
        previous = neighbour_score.get(neighbour_id)
        if previous is None or confidence > previous:
            neighbour_score[neighbour_id] = confidence

    if not neighbour_score:
        return []

    meta_q = (
        build_base_query(
            user_id=user_id,
            is_admin=is_admin,
            allowed_domains=allowed_domains,
            container_id=container_id,
        )
        .where(FileMetadata.file_id.in_(list(neighbour_score.keys())))
    )
    meta_rows = (await db.execute(meta_q)).scalars().all()

    results = [(meta, neighbour_score[meta.file_id]) for meta in meta_rows]
    results.sort(key=lambda item: item[1], reverse=True)
    return results[:limit]
