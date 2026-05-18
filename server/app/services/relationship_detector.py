"""Relationship detection between ingested files using a key fingerprint index.

WHY THE OLD APPROACH WAS WRONG
===============================
Old approach:
    SELECT * FROM file_metadata WHERE file_id != :this_id   ← full table scan
    for each other file:
        for each column in this file:
            if column_name.lower() in other_file_columns:   ← string equality
                create relationship

Problems:
1. O(N²) — at 100k files, 10 billion comparisons per upload. DB dies.
2. String equality misses equivalent source-system and business identifiers.
3. N individual SELECT queries per column match → N extra DB round trips.
4. Confidence formula (0.5 + overlap * 0.5) is arbitrary — no semantic meaning.
5. "id" in two different files creates false positive join with confidence 0.5.

NEW APPROACH — registry + fingerprint evidence
==============================================
After role resolution (column_role_resolver.py), each file has typed roles:
    column_semantic_roles = {"source_col": "custom:entity_key:record"}

Key-like roles are written into column_key_registry with normalized value
fingerprints. Relationship detection uses the registry's tenant-scoped GIN
array overlap index. Role labels decide which columns are eligible; actual
value overlap is required before a FileRelationship is created.

The detector intentionally avoids creating joins from role names alone. At
million-file scale, a shared label is a candidate signal, not proof.
"""
from __future__ import annotations

import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import ingest_logger
from app.models.file_metadata import FileMetadata
from app.models.file_relationship import FileRelationship
from app.services.relationship_index import (
    find_fingerprint_matches,
    register_file_key_fingerprints,
)
from app.services.semantic_policy import get_semantic_policy
from app.services.semantic_roles import is_relationship_role


def _canonical_match(match: dict) -> dict:
    """Return a deterministic relationship orientation for pair dedupe.

    In normal ingestion only the newly ingested file is checked, so inverse
    duplicates are unlikely. A container-wide semantic rebuild checks every file;
    deterministic ordering keeps A->B and B->A from becoming two edges.
    """
    file_a_id = str(match["file_a_id"])
    file_b_id = str(match["file_b_id"])
    if file_a_id <= file_b_id:
        return {
            "file_a_id": file_a_id,
            "file_b_id": file_b_id,
            "file_a_path": match.get("file_a_path"),
            "file_b_path": match.get("file_b_path"),
            "col_a": match.get("col_a"),
            "col_b": match.get("col_b"),
            "role": match.get("role_a") or match.get("role_b"),
        }
    return {
        "file_a_id": file_b_id,
        "file_b_id": file_a_id,
        "file_a_path": match.get("file_b_path"),
        "file_b_path": match.get("file_a_path"),
        "col_a": match.get("col_b"),
        "col_b": match.get("col_a"),
        "role": match.get("role_b") or match.get("role_a"),
    }


async def detect_relationships(
    file_id: str,
    blob_path: str,
    columns_info: list[dict],
    db: AsyncSession,
) -> int:
    """Detect relationships between the newly ingested file and all prior files.

    Uses role-indexed PostgreSQL query (GIN index, O(log N)).
    Also registers this file in the database-backed column fingerprint index and
    uses value overlap to confirm relationships. No runtime O(N²) scan.

    Returns count of new FileRelationship rows created.
    """
    policy = get_semantic_policy()

    # Load this file's metadata to get resolved roles
    this_meta_result = await db.execute(
        select(FileMetadata).where(FileMetadata.file_id == file_id)
    )
    this_meta = this_meta_result.scalar_one_or_none()
    this_roles: dict[str, str] = (this_meta.column_semantic_roles or {}) if this_meta else {}

    created = 0

    # Register the current file's candidate key columns in the durable inverted
    # index. This is the database-backed hashmap: value fingerprint -> columns.
    # It is scoped by container_id in every lookup, so tenants never mix.
    try:
        await register_file_key_fingerprints(file_id, db)
    except Exception as exc:
        ingest_logger.warning(
            "column_key_registry_failed",
            file_id=file_id,
            error=str(exc)[:300],
        )

    # ── Value-overlap path: data-backed relationships ────────────────────────
    # This is stronger than semantic role matching.  It verifies that actual key
    # values overlap using a GIN-indexed array lookup on column_key_registry.
    try:
        matches = await find_fingerprint_matches(file_id, db)
    except Exception as exc:
        matches = []
        ingest_logger.warning(
            "relationship_detector",
            mode="fingerprint_index",
            status="failed",
            error=str(exc)[:300],
        )

    for match in matches:
        overlap_pct = float(match["overlap_pct"] or 0.0)
        if overlap_pct < policy.min_value_overlap:
            continue

        canonical = _canonical_match(match)
        role = canonical["role"]
        confidence = min(policy.fingerprint_max_confidence, max(policy.fingerprint_min_confidence, overlap_pct))
        join_type = "INNER JOIN" if overlap_pct >= policy.inner_join_overlap else "LEFT JOIN"

        existing = await db.execute(
            select(FileRelationship).where(
                FileRelationship.file_a_id == canonical["file_a_id"],
                FileRelationship.file_b_id == canonical["file_b_id"],
                FileRelationship.shared_column == canonical["col_a"],
            )
        )
        rel = existing.scalar_one_or_none()
        if not rel:
            rel = FileRelationship(
                id=str(uuid.uuid4()),
                file_a_id=canonical["file_a_id"],
                file_b_id=canonical["file_b_id"],
                file_a_path=canonical["file_a_path"],
                file_b_path=canonical["file_b_path"],
                shared_column=canonical["col_a"],
                related_column=canonical["col_b"],
                semantic_role=role,
                role_source="fingerprint_index",
                confidence_score=confidence,
                value_overlap_pct=overlap_pct,
                join_type=join_type,
            )
            db.add(rel)
            created += 1
        else:
            rel.confidence_score = max(rel.confidence_score, confidence)
            rel.semantic_role = rel.semantic_role or role
            rel.related_column = rel.related_column or canonical["col_b"]
            rel.role_source = "fingerprint_index"
            rel.value_overlap_pct = max(rel.value_overlap_pct or 0.0, overlap_pct)
            rel.join_type = "INNER JOIN" if (rel.value_overlap_pct or 0.0) >= policy.inner_join_overlap else "LEFT JOIN"

    if matches:
        ingest_logger.info(
            "relationship_detector",
            mode="fingerprint_index",
            candidates=len(matches),
            relationships_created=created,
        )

    key_roles = sorted({role for role in this_roles.values() if is_relationship_role(role)})
    ingest_logger.info(
        "relationship_detector",
        mode="role_registry",
        key_roles=key_roles,
        note="role labels drive key registration; relationships require value-overlap evidence",
    )

    await db.commit()
    ingest_logger.info(
        "relationship_detector",
        file_id=file_id,
        relationships_created=created,
    )
    return created

