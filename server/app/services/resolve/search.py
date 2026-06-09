"""Per-entity candidate retrieval for the brain-resolve seam.

Given the decomposed entities, this module retrieves a SMALL, twin-aware slice of
candidate tables PER ENTITY — the narrow set the brain then reasons over. It is
the SEARCH half of the seam:

  * One vector retrieval per entity (cosine over ``description_embedding``,
    scoped to the container) so each business object the question spans is
    resolved independently rather than blended into one ranked list.
  * TWINS-TOGETHER: every schema-fingerprint sibling of a hit is pulled in, so a
    genuine canonical-vs-lookalike decision is never lost because one twin
    out-ranked the others on description text. Twins are grouped by
    ``erp_classifications.schema_fingerprint`` (FileMetadata has no fingerprint
    column).

Each returned candidate is ``{"file_id", "table"}`` where ``table`` is the
LOGICAL table name the executor uses — resolved from the SAME
``file_identity_map`` the executor trusts (``.by_id[file_id].logical_name``),
falling back to ``logical_table_key(blob_path)`` only when the map lacks the id.

Design properties (enforced):
  * Container-scoped, read-only. No cross-file conclusions are computed here; the
    twin grouping is a precomputed-fingerprint lookup, not an all-pairs join.
  * Never raises. On any failed query the session is rolled back and whatever was
    gathered so far is returned, so the seam stays additive and falls through.
"""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.erp_classification import ErpClassification
from app.models.file_metadata import FileMetadata
from app.retrieval.embeddings import embed_text
from app.services.file_identity import FileIdentityMap, logical_table_key

logger = structlog.get_logger("resolve.search")


def _is_zero_vector(vec: list[float]) -> bool:
    """True when embed_text returned a zero vector (Azure offline / degraded)."""
    return not vec or all(v == 0.0 for v in vec)


def _logical_table(
    file_id: str, blob_path: str | None, file_identity_map: FileIdentityMap | None,
) -> str | None:
    """Resolve the logical table name the executor uses for a file.

    Prefer the request's identity map (``.by_id[file_id].logical_name``) so the
    name matches exactly what canonicalize_logical_sql will resolve; fall back to
    the lexical ``logical_table_key`` only when the map lacks the id."""
    if file_identity_map is not None:
        identity = getattr(file_identity_map, "by_id", {}).get(file_id)
        if identity is not None and getattr(identity, "logical_name", None):
            return identity.logical_name
    if blob_path:
        return logical_table_key(blob_path)
    return None


async def _embed(entity: str) -> list[float] | None:
    try:
        vec = await embed_text(entity.strip())
    except Exception as exc:  # noqa: BLE001 — never raise
        logger.warning("embed_error", entity=entity[:80], error=str(exc)[:160])
        return None
    if _is_zero_vector(vec):
        logger.info("embed_zero_vector", entity=entity[:80])
        return None
    return vec


async def _vector_hits(
    db: AsyncSession,
    container_id: str,
    vec: list[float],
    top_k: int,
    min_score: float,
) -> list[tuple[str, str | None, float]]:
    """Container-scoped cosine search over description_embedding.

    Returns ``[(file_id, blob_path, similarity), ...]`` best-first. A direct
    pgvector ``cosine_distance`` query (rather than the retrieval
    ``vector_search``) keeps this scoped to container_id alone — authorization is
    enforced downstream by the executor's allowed_file_ids allowlist, so the
    permission machinery is not double-applied here."""
    distance = FileMetadata.description_embedding.cosine_distance(vec)
    similarity = (1.0 - distance).label("similarity")
    q = (
        select(FileMetadata.file_id, FileMetadata.blob_path, similarity)
        .where(FileMetadata.container_id == container_id)
        .where(FileMetadata.description_embedding.is_not(None))
        .order_by(distance.asc())
        .limit(top_k)
    )
    try:
        rows = (await db.execute(q)).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("vector_query_error", error=str(exc)[:200])
        return []
    return [
        (str(fid), bp, float(sim))
        for fid, bp, sim in rows
        if fid and float(sim) >= min_score
    ]


async def _twin_siblings(
    db: AsyncSession,
    container_id: str,
    hit_file_ids: list[str],
) -> dict[str, list[str]]:
    """For the hit files, find their schema-twin siblings.

    Groups by ``erp_classifications.schema_fingerprint`` so all members of a
    twin cluster are kept together. Returns ``{fingerprint: [file_id, ...]}`` for
    the fingerprints the hits belong to. Read of a precomputed column — not an
    all-pairs comparison."""
    if not hit_file_ids:
        return {}
    try:
        # Fingerprints of the hit files.
        fp_rows = (
            await db.execute(
                select(ErpClassification.file_id, ErpClassification.schema_fingerprint)
                .where(ErpClassification.container_id == container_id)
                .where(ErpClassification.file_id.in_(hit_file_ids))
            )
        ).all()
        fingerprints = {fp for _fid, fp in fp_rows if fp}
        if not fingerprints:
            return {}
        # All container files that share those fingerprints (the twin clusters).
        sib_rows = (
            await db.execute(
                select(ErpClassification.file_id, ErpClassification.schema_fingerprint)
                .where(ErpClassification.container_id == container_id)
                .where(ErpClassification.schema_fingerprint.in_(list(fingerprints)))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("twin_query_error", error=str(exc)[:200])
        return {}
    clusters: dict[str, list[str]] = {}
    for fid, fp in sib_rows:
        if fp and fid:
            clusters.setdefault(fp, []).append(str(fid))
    return clusters


async def _blob_paths_for(
    db: AsyncSession, file_ids: list[str],
) -> dict[str, str | None]:
    """Bulk-load blob_path for the given file ids (twin siblings may not have
    been in the vector hit set, so their blob_path is unknown until now)."""
    if not file_ids:
        return {}
    try:
        rows = (
            await db.execute(
                select(FileMetadata.file_id, FileMetadata.blob_path)
                .where(FileMetadata.file_id.in_(file_ids))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("blob_path_query_error", error=str(exc)[:200])
        return {}
    return {str(fid): bp for fid, bp in rows}


async def search_per_entity(
    db: AsyncSession,
    container_id: str,
    entities: list[str],
    top_k: int = 9,
    min_score: float = 0.0,
    file_identity_map: FileIdentityMap | None = None,
) -> dict[str, list[dict]]:
    """Retrieve a twin-aware candidate slice per entity.

    For each entity: embed it, cosine-search the container's
    ``description_embedding`` (top_k, above ``min_score``), then pull every
    schema-twin sibling of each hit so the canonical-vs-lookalike decision is
    intact. Returns ``{entity: [{"file_id", "table"}, ...]}`` (best-first hits
    first, then twins), capped at ``top_k + twins``. Never raises.
    """
    results: dict[str, list[dict]] = {}
    if not entities or not container_id:
        return results

    for entity in entities:
        if not entity or not entity.strip():
            continue
        vec = await _embed(entity)
        if vec is None:
            results[entity] = []
            continue

        hits = await _vector_hits(db, container_id, vec, top_k, min_score)
        ordered_ids: list[str] = [fid for fid, _bp, _sim in hits]
        blob_by_id: dict[str, str | None] = {fid: bp for fid, bp, _sim in hits}

        # Twins-together: expand each hit with its fingerprint siblings.
        clusters = await _twin_siblings(db, container_id, ordered_ids)
        twin_ids: list[str] = []
        for members in clusters.values():
            for fid in members:
                if fid not in blob_by_id and fid not in twin_ids:
                    twin_ids.append(fid)
        if twin_ids:
            twin_blobs = await _blob_paths_for(db, twin_ids)
            blob_by_id.update(twin_blobs)

        # Hits first (rank order), then twins. Cap at top_k + twin count so the
        # slice stays small even if a cluster is large.
        candidate_ids = ordered_ids + [fid for fid in twin_ids if fid not in ordered_ids]
        cap = top_k + len(twin_ids)
        candidates: list[dict] = []
        seen: set[str] = set()
        for fid in candidate_ids[:cap]:
            if fid in seen:
                continue
            seen.add(fid)
            table = _logical_table(fid, blob_by_id.get(fid), file_identity_map)
            if not table:
                continue
            candidates.append({"file_id": fid, "table": table})

        results[entity] = candidates
        logger.info(
            "search_entity",
            entity=entity[:80],
            hits=len(hits),
            twins=len(twin_ids),
            candidates=len(candidates),
        )

    return results
