"""[3b] ASSEMBLE — build the per-file EVIDENCE packet for a candidate slice.

Given a twin-aware ``CandidateSlice`` (from [3a] LOOKUP), assemble the clean,
per-file EVIDENCE the proposer reasons over and the verifier value-checks against:
typed ``columns`` + semantic roles, a few real ``sample_rows``, the stored
categorical ``value_set`` (UPPER(col) -> {value: count}), per-column
``unique_rates``, and the business-context discriminator (``polarity`` /
``process_role`` / ``erp_module``). All of it is per-file EVIDENCE that ingestion
already stored — no cross-file conclusions are computed here (INVARIANT I13).

This module is SELF-CONTAINED: the assembler is LIFTED here (the navigator must not
import from ``app.services.resolve.*``, which is deleted at P5). It reads exactly the
tables ingestion populates — FileMetadata, FileAnalytics, ColumnKeyRegistry,
ErpClassification — and applies the classifier's own polarity-reliability gate via
the navigator-local ``verifier._polarity_from_row``.

Defensive: NEVER raises. Every DB read rolls back and degrades on error; the whole
``assemble`` call is additionally guarded so an empty packet is returned rather than
propagating an exception into the loop.
"""
from __future__ import annotations

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.column_key_registry import ColumnKeyRegistry
from app.models.erp_classification import ErpClassification
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.services.navigator.types import CandidateSlice, EvidencePacket
# Self-contained: the polarity-reliability gate is the navigator-local copy in the
# verifier (no resolve.* import).
from app.services.navigator.verifier import _polarity_from_row

logger = structlog.get_logger("navigator.evidence")

_SAMPLE_ROWS = 3          # real rows per candidate — enough to read the data, cheap
_MAX_SLICE = 9            # match the per-step search top_k: see every genuine hit


def _slice_to_catalog(slice: CandidateSlice) -> list[dict]:
    """Project the slice's candidates into the ``[{"file_id", "table"}, ...]``
    catalog shape the assembler consumes. Pure."""
    catalog: list[dict] = []
    for c in slice.candidates:
        if c.file_id and c.table:
            catalog.append({"file_id": c.file_id, "table": c.table})
    return catalog


async def assemble_evidence(db: AsyncSession, candidates: list[dict]) -> list[dict]:
    """Build the clean per-file EVIDENCE for the candidate slice. LIFTED from the
    legacy seam so the navigator is self-contained.

    ``candidates`` are catalog entries (each carrying ``file_id`` and the logical
    ``table`` name the SQL must use). For each we pull the per-file evidence the
    brain reasons over: typed columns + semantic roles + a few real sample rows +
    description + value_set + unique_rates + business-context. No conclusions — only
    what ingestion stored per file.
    """
    by_file: dict[str, dict] = {}
    for c in candidates[:_MAX_SLICE]:
        fid = c.get("file_id")
        if fid:
            by_file[fid] = c
    if not by_file:
        return []
    file_ids = list(by_file.keys())
    # FileMetadata is the spine of the packet — guard it: a read failure here can't
    # be degraded (there's nothing to build a packet without), so roll back and
    # return [] (mirrors the ErpClassification guard's rollback-and-degrade pattern).
    try:
        rows = (
            await db.execute(
                select(
                    FileMetadata.file_id,
                    FileMetadata.columns_info,
                    FileMetadata.column_semantic_roles,
                    FileMetadata.sample_rows,
                    FileMetadata.ai_description,
                    FileMetadata.good_for,
                    FileMetadata.date_range_start,
                    FileMetadata.date_range_end,
                    FileMetadata.row_count,
                ).where(FileMetadata.file_id.in_(file_ids))
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("file_metadata_read_error", error=str(exc)[:200])
        return []

    # Stored value-sets + key-registry unique-rates per file, used by verify() to
    # value-check filters and sanity-check the grain. Both reads are defensive: a
    # missing/None entry means "can't disprove" → verify passes (abstain-biased).
    # Each read is independently guarded: a transient blip on value_counts /
    # unique_rate must DEGRADE that one signal to {} and NOT discard the columns /
    # roles that already loaded fine (the docstring's per-read promise).
    value_counts_by_file: dict[str, dict] = {}
    try:
        for fid, vc in (
            await db.execute(
                select(FileAnalytics.file_id, FileAnalytics.value_counts)
                .where(FileAnalytics.file_id.in_(file_ids))
            )
        ).all():
            if isinstance(vc, dict):
                value_counts_by_file[fid] = vc
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("value_counts_read_error", error=str(exc)[:200])
        value_counts_by_file = {}

    unique_rate_by_file: dict[str, dict[str, float]] = {}
    try:
        for fid, cname, urate in (
            await db.execute(
                select(
                    ColumnKeyRegistry.file_id,
                    ColumnKeyRegistry.column_name,
                    ColumnKeyRegistry.unique_rate,
                ).where(ColumnKeyRegistry.file_id.in_(file_ids))
            )
        ).all():
            if cname is not None:
                unique_rate_by_file.setdefault(fid, {})[str(cname).upper()] = float(urate or 0.0)
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("unique_rate_read_error", error=str(exc)[:200])
        unique_rate_by_file = {}

    # <seam: KB-build Task 6 adds a declared-metrics read after this block>

    # ERP business-context classification per file (the twin/AP-vs-AR discriminator).
    # Read defensively — gated, rollback-safe, missing row ⇒ polarity=None. The
    # reliability gate (_polarity_from_row) is applied at read time.
    polarity_by_file: dict[str, str | None] = {}
    role_by_file: dict[str, str] = {}
    module_by_file: dict[str, str] = {}
    conf_by_file: dict[str, float] = {}
    if getattr(get_settings(), "BRAIN_POLARITY_GATE_ENABLED", True):
        try:
            for fid, polarity, role, module, conf, src, src_sys in (
                await db.execute(
                    select(
                        ErpClassification.file_id,
                        ErpClassification.domain_polarity,
                        ErpClassification.process_role,
                        ErpClassification.erp_module,
                        ErpClassification.confidence,
                        ErpClassification.source,
                        ErpClassification.source_system,
                    ).where(ErpClassification.file_id.in_(file_ids))
                )
            ).all():
                polarity_by_file[fid] = _polarity_from_row(polarity, conf, src, src_sys)
                role_by_file[fid] = str(role or "")
                module_by_file[fid] = str(module or "")
                try:
                    conf_by_file[fid] = float(conf or 0.0)
                except (TypeError, ValueError):
                    conf_by_file[fid] = 0.0
        except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
            await db.rollback()
            logger.warning("erp_classification_read_error", error=str(exc)[:200])

    evidence: list[dict] = []
    for fid, columns_info, roles, sample_rows, descr, good_for, d0, d1, row_count in rows:
        cat = by_file.get(fid, {})
        table = cat.get("table") or cat.get("logical_name") or cat.get("display_name")
        if not table:
            continue
        roles = roles or {}
        cols: list[dict] = []
        valid: dict[str, str] = {}          # upper -> exact-case, for verification
        for ci in (columns_info or []):
            cname = ci.get("name") if isinstance(ci, dict) else str(ci)
            if not cname:
                continue
            valid[str(cname).upper()] = str(cname)
            cols.append({
                "name": cname,
                "type": (ci.get("dtype") or ci.get("type") or "") if isinstance(ci, dict) else "",
                "role": roles.get(cname, ""),
            })
        # value_counts keyed by UPPER(column) for case-insensitive lookup in verify().
        raw_vc = value_counts_by_file.get(fid) or {}
        value_set: dict[str, dict] = {
            str(k).upper(): v for k, v in raw_vc.items() if isinstance(v, dict)
        }
        evidence.append({
            "table": table,
            "valid_cols": valid,
            "columns": cols,
            "sample_rows": (sample_rows or [])[:_SAMPLE_ROWS],
            "description": (descr or "")[:400],
            "good_for": (good_for or [])[:6],
            "coverage": f"{d0 or '?'}..{d1 or '?'}",
            "row_count": int(row_count or 0),
            "value_set": value_set,                       # UPPER(col) -> {value: count}
            "unique_rates": unique_rate_by_file.get(fid, {}),  # UPPER(col) -> unique_rate
            # Business-context discriminator (gated; None when off/unreliable/missing).
            "polarity": polarity_by_file.get(fid),         # customer|vendor|None
            "process_role": role_by_file.get(fid, ""),
            "erp_module": module_by_file.get(fid, ""),
            "erp_confidence": conf_by_file.get(fid, 0.0),
        })
    return evidence


async def assemble(db: AsyncSession, slice: CandidateSlice) -> EvidencePacket:
    """Assemble the EVIDENCE packet for ``slice`` — one evidence dict per candidate
    file carrying columns / roles / sample_rows / value_set / unique_rates /
    polarity. NEVER raises (empty packet on any failure or empty slice).
    """
    step_id = getattr(slice, "step_id", "") if slice is not None else ""
    if slice is None or not slice.candidates:
        return EvidencePacket(step_id=step_id, files=())

    catalog = _slice_to_catalog(slice)
    if not catalog:
        return EvidencePacket(step_id=step_id, files=())

    try:
        files = await assemble_evidence(db, catalog)
    except Exception as exc:  # noqa: BLE001 — never raise; degrade to empty packet
        try:
            await db.rollback()
        except Exception:  # noqa: BLE001
            pass
        logger.warning("assemble_error", step_id=step_id, error=str(exc)[:200])
        return EvidencePacket(step_id=step_id, files=())

    logger.info("assemble_ok", step_id=step_id, n_files=len(files or []))
    return EvidencePacket(step_id=step_id, files=tuple(files or ()))
