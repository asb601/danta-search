"""Database-backed column fingerprint registry for relationship discovery.

This is the production version of the user's hashmap idea:
- not process memory
- not global across tenants
- indexed in PostgreSQL with GIN array overlap
- populated at ingestion time from sample rows + semantic roles
"""
from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import ingest_logger
from app.models.column_key_registry import ColumnKeyRegistry
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.services.ingestion_config import glossary_filename_tokens, null_tokens_lower
from app.services.semantic_policy import SemanticPolicy, get_semantic_policy
from app.services.semantic_roles import (
    is_fingerprint_key_role,
    is_never_fingerprint_join_role,
)

_NULL_LIKE = null_tokens_lower()
_LEADING_ZERO_INT_RE = re.compile(r"^0+(\d+)$")


def is_dictionary_like_path(name_or_path: str | None) -> bool:
    """Return True for configured schema/glossary/data-dictionary files."""
    stem = Path(str(name_or_path or "")).stem.lower()
    return bool(stem and any(token in stem for token in glossary_filename_tokens()))


@dataclass(frozen=True)
class RegistryCandidate:
    column_name: str
    semantic_role: str | None
    key_kind: str
    values: list[str]
    sample_size: int
    null_rate: float
    unique_rate: float


def normalize_key_value(value: Any) -> str | None:
    """Normalize business-key values before hashing.

    Keeps the rule generic: trim, lowercase, collapse spaces, normalize numeric
    identifiers with leading zeros (000123 -> 123). No client-specific aliases.
    """
    if value is None:
        return None
    text_value = str(value).strip().lower()
    if text_value in _NULL_LIKE:
        return None
    text_value = re.sub(r"\s+", " ", text_value)
    match = _LEADING_ZERO_INT_RE.match(text_value)
    if match:
        text_value = match.group(1)
    return text_value or None


def fingerprint_value(value: Any) -> str | None:
    normalized = normalize_key_value(value)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _values_for_column(
    sample_rows: list[dict],
    column_name: str,
    max_fingerprints: int,
) -> tuple[list[str], int, int]:
    seen: set[str] = set()
    values: list[str] = []
    null_count = 0
    sample_size = 0

    for row in sample_rows or []:
        if column_name not in row:
            continue
        sample_size += 1
        normalized = normalize_key_value(row.get(column_name))
        if normalized is None:
            null_count += 1
            continue
        if normalized not in seen:
            seen.add(normalized)
            values.append(normalized)
            if len(values) >= max_fingerprints:
                break

    return values, sample_size, null_count


def _candidate_kind(
    role: str | None,
    unique_rate: float,
    null_rate: float,
    policy: SemanticPolicy,
) -> str | None:
    if is_never_fingerprint_join_role(role):
        return None

    # Ontology says this is a business key. Accept it unless it is too null or degenerate.
    if is_fingerprint_key_role(role):
        if null_rate <= policy.max_join_null_rate and unique_rate > policy.ontology_key_min_unique_rate:
            return "pk" if unique_rate >= policy.pk_unique_rate and null_rate <= policy.pk_null_rate else "fk"
        return None

    # Role missing: use conservative generic cardinality signals only.
    if null_rate <= policy.pk_null_rate and unique_rate >= policy.generic_pk_unique_rate:
        return "pk"
    if (
        null_rate <= policy.generic_fk_max_null_rate
        and policy.generic_fk_min_unique_rate <= unique_rate <= policy.generic_fk_max_unique_rate
    ):
        return "fk"
    return None


def build_registry_candidates(meta: FileMetadata) -> list[RegistryCandidate]:
    policy = get_semantic_policy()
    roles: dict[str, str] = meta.column_semantic_roles or {}
    sample_rows: list[dict] = meta.sample_rows or []
    candidates: list[RegistryCandidate] = []

    for col in meta.columns_info or []:
        column_name = col.get("name") if isinstance(col, dict) else None
        if not column_name:
            continue

        values, sample_size, null_count = _values_for_column(
            sample_rows,
            column_name,
            policy.max_fingerprints_per_column,
        )
        if len(values) < policy.min_distinct_key_values or sample_size <= 0:
            continue

        null_rate = null_count / sample_size
        unique_rate = len(values) / max(sample_size - null_count, 1)
        role = roles.get(column_name)
        key_kind = _candidate_kind(role, unique_rate, null_rate, policy)
        if not key_kind:
            continue

        candidates.append(RegistryCandidate(
            column_name=column_name,
            semantic_role=role,
            key_kind=key_kind,
            values=values,
            sample_size=sample_size,
            null_rate=null_rate,
            unique_rate=unique_rate,
        ))

    return candidates


async def register_file_key_fingerprints(file_id: str, db: AsyncSession) -> int:
    """Upsert all candidate key columns for one file into column_key_registry."""
    file = await db.get(File, file_id)
    if file and is_dictionary_like_path(file.name):
        await db.execute(delete(ColumnKeyRegistry).where(ColumnKeyRegistry.file_id == file_id))
        await db.commit()
        ingest_logger.info(
            "column_key_registry_skipped",
            file_id=file_id,
            reason="dictionary_file_not_joinable",
        )
        return 0

    result = await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
    meta = result.scalar_one_or_none()
    if not meta or not meta.container_id:
        return 0

    candidates = build_registry_candidates(meta)
    policy = get_semantic_policy()

    await db.execute(delete(ColumnKeyRegistry).where(ColumnKeyRegistry.file_id == file_id))
    inserted = 0
    for candidate in candidates:
        fps = sorted({fp for v in candidate.values if (fp := fingerprint_value(v))})
        if len(fps) < policy.min_distinct_key_values:
            continue
        db.add(ColumnKeyRegistry(
            id=str(uuid.uuid4()),
            container_id=meta.container_id,
            file_id=meta.file_id,
            blob_path=meta.blob_path,
            column_name=candidate.column_name,
            semantic_role=candidate.semantic_role,
            key_kind=candidate.key_kind,
            cardinality=len(fps),
            sample_size=candidate.sample_size,
            unique_rate=candidate.unique_rate,
            null_rate=candidate.null_rate,
            value_fingerprints=fps,
        ))
        inserted += 1

    await db.commit()
    ingest_logger.info(
        "column_key_registry_updated",
        file_id=file_id,
        candidates=len(candidates),
        inserted=inserted,
    )
    return inserted


async def find_fingerprint_matches(file_id: str, db: AsyncSession) -> list[dict]:
    """Find candidate relationships for one file using GIN array overlap.

    All matches are constrained to the same container_id. This prevents tenant or
    organization cross-talk even when fingerprints overlap globally.
    """
    sql = text(
        """
        WITH this_cols AS (
            SELECT *
            FROM column_key_registry
            WHERE file_id = :file_id
        )
        SELECT
            this_cols.file_id AS file_a_id,
            this_cols.blob_path AS file_a_path,
            this_cols.column_name AS col_a,
            this_cols.semantic_role AS role_a,
            this_cols.key_kind AS key_kind_a,
            this_cols.cardinality AS card_a,
            other.file_id AS file_b_id,
            other.blob_path AS file_b_path,
            other.column_name AS col_b,
            other.semantic_role AS role_b,
            other.key_kind AS key_kind_b,
            other.cardinality AS card_b,
            ARRAY(
                SELECT UNNEST(this_cols.value_fingerprints)
                INTERSECT
                SELECT UNNEST(other.value_fingerprints)
            ) AS overlap_values
        FROM this_cols
        JOIN column_key_registry other
          ON other.container_id = this_cols.container_id
         AND other.file_id != this_cols.file_id
         AND other.value_fingerprints && this_cols.value_fingerprints
        """
    )
    rows = (await db.execute(sql, {"file_id": file_id})).mappings().all()
    policy = get_semantic_policy()
    matches: list[dict] = []
    for row in rows:
        if is_dictionary_like_path(row["file_a_path"]) or is_dictionary_like_path(row["file_b_path"]):
            continue
        overlap_count = len(row["overlap_values"] or [])
        if overlap_count < policy.min_overlap_fingerprint_count:
            continue
        min_cardinality = max(min(row["card_a"] or 0, row["card_b"] or 0), 1)
        overlap_pct = overlap_count / min_cardinality
        if overlap_pct < policy.min_value_overlap:
            continue
        matches.append({**dict(row), "overlap_count": overlap_count, "overlap_pct": overlap_pct})
    matches.sort(key=lambda item: item["overlap_pct"], reverse=True)
    return matches
