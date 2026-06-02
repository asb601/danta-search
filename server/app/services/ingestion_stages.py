"""Staged ingestion operations used by Celery.

Each function is one durable ingestion stage. Celery owns orchestration and
retry; this module owns database and artifact mutations for that stage.
"""
from __future__ import annotations

import time
import uuid
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ai_client import generate_file_description
from app.core.database import async_session as _async_session
from app.core.db_logger import log_ingest_event as _db_log
from app.core.duckdb_client import sample_file
from app.core.logger import ingest_logger
from app.models.container import ContainerConfig
from app.models.file import File
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder
from app.retrieval.embeddings import build_search_text, embed_text
from app.services.analytics_service import compute_and_store_analytics, trigger_parquet_conversion
from app.services.data_preprocessor import preprocess_file
from app.services.ingestion_config import (
    IngestStatus,
    PayloadStatus,
    StageName,
    is_parquet_source_file,
    preprocess_extensions,
)
from app.services.ingestion_service import _is_schema_file, _load_schema_glossary

Payload = dict[str, Any]
_PREPROCESS_EXTS = preprocess_extensions(dotted=True)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _file_ext(file: File) -> str:
    return Path(file.name).suffix.lower()


def _escalation_policy() -> tuple[bool, float]:
    """Return (escalation_enabled, confidence_threshold) from ingestion policy.

    Defaults to (False, 0.55). When disabled, every escalation hook below is a
    no-op and behaviour is byte-identical to today.
    """
    try:
        from app.services.ingestion_policy import get_ingestion_policy  # noqa: PLC0415

        pol = get_ingestion_policy()
        enabled = bool(pol.lookup(("model_pool", "escalation_enabled")))
        raw_threshold = pol.lookup(("model_pool", "escalation_confidence_threshold"))
        threshold = float(raw_threshold) if raw_threshold is not None else 0.55
    except Exception:  # noqa: BLE001 — policy failure must never block ingestion
        return False, 0.55
    return enabled, threshold


def _next(payload: Payload, *, stage: str, **extra: Any) -> Payload:
    merged = dict(payload)
    merged.update(extra)
    merged["stage"] = stage
    return merged


async def _resolve_org_ai(db: AsyncSession, container: ContainerConfig) -> dict[str, Any] | None:
    """Resolve per-org AI settings for a container's organization.

    Returns the org_ai dict from resolve_org_ai_settings, or None on any
    failure / when no organization is associated — None means "use the global
    AI path", which is byte-identical to the legacy behaviour.
    """
    try:
        org_id = getattr(container, "organization_id", None)
        if not org_id:
            return None
        from app.services.org_ai_resolver import resolve_org_ai_settings  # noqa: PLC0415

        return await resolve_org_ai_settings(org_id, db)
    except Exception:  # noqa: BLE001 — resolution must never block ingestion
        return None


async def _load_file_and_container(db: AsyncSession, file_id: str) -> tuple[File, ContainerConfig]:
    file = await db.get(File, file_id)
    if not file or not file.blob_path:
        raise RuntimeError("file or blob_path missing")

    container = await db.get(ContainerConfig, file.container_id)
    if not container:
        raise RuntimeError("container not found")

    return file, container


async def _blob_exists(connection_string: str, container_name: str, blob_path: str) -> bool:
    def _check() -> bool:
        try:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415

            client = BlobServiceClient.from_connection_string(connection_string)
            return client.get_blob_client(container=container_name, blob=blob_path).exists()
        except Exception:
            return False

    import asyncio

    return await asyncio.to_thread(_check)


async def prepare_pipeline(file_id: str) -> Payload:
    """Mark a file as queued for staged ingestion.

    This is intentionally small and fast. It prevents duplicate UI clicks from
    queuing a second chain while a file is already pending.
    """
    async with _async_session() as db:
        file = await db.get(File, file_id)
        if not file or not file.blob_path:
            return {"file_id": file_id, "status": PayloadStatus.SKIPPED.value, "reason": "file or blob_path missing"}

        if file.ingest_status == IngestStatus.RUNNING.value:
            return {"file_id": file_id, "status": PayloadStatus.ALREADY_RUNNING.value}

        # Resolve actor fields for RBAC-aware log rows
        from app.models.user import User  # local import to avoid circular
        owner_id = file.uploaded_by_id or file.owner_id
        actor: User | None = await db.get(User, owner_id) if owner_id else None

        file.ingest_status = IngestStatus.RUNNING.value
        await db.commit()

        payload_out = {
            "file_id": file_id,
            "status": PayloadStatus.QUEUED.value,
            "actor_user_id": str(actor.id) if actor else None,
            "actor_email": actor.email if actor else None,
            "actor_role": actor.role if actor else None,
        }
        await _db_log(
            event="chain_start",
            level="info",
            trace_id=file_id,
            file_id=file_id,
            file_name=file.name,
            actor_user_id=payload_out["actor_user_id"],
            actor_email=payload_out["actor_email"],
            actor_role=payload_out["actor_role"],
            details={"filename": file.name, "blob_path": file.blob_path},
        )
        return payload_out


async def clean_file_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.CLEAN.value

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        ext = _file_ext(file)

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)

        if ext in _PREPROCESS_EXTS and not file.is_preprocessed:
            analytics_row = (
                await db.execute(select(FileAnalytics).where(FileAnalytics.file_id == file_id))
            ).scalar_one_or_none()
            if analytics_row and analytics_row.parquet_blob_path:
                analytics_row.parquet_blob_path = None
                analytics_row.parquet_size_bytes = None
                await db.commit()

            prep = await preprocess_file(
                blob_path=file.blob_path,
                file_name=file.name,
                file_id=file_id,
                connection_string=container.connection_string,
                container_name=container.container_name,
                cleaning_config=container.cleaning_config,
            )
            file.blob_path = prep.clean_blob_path
            file.is_preprocessed = True

            analytics_row = (
                await db.execute(select(FileAnalytics).where(FileAnalytics.file_id == file_id))
            ).scalar_one_or_none()
            if not analytics_row:
                analytics_row = FileAnalytics(id=str(uuid.uuid4()), file_id=file_id)
                db.add(analytics_row)
            analytics_row.quarantine_count = prep.quarantine_count
            analytics_row.quarantine_sample = prep.quarantine_sample
            analytics_row.cleaning_audit = prep.cleaning_audit
            await db.commit()

            dur = _ms(start)
            ingest_logger.info(
                "ingest_stage",
                stage=stage,
                status="done",
                file_id=file_id,
                clean_blob_path=prep.clean_blob_path,
                original_rows=prep.original_rows,
                clean_rows=prep.clean_rows,
                duration_ms=dur,
            )
            await _db_log(
                event="ingest_stage",
                level="info",
                trace_id=file_id,
                file_id=file_id,
                file_name=file.name,
                duration_ms=dur,
                actor_user_id=payload.get("actor_user_id"),
                actor_email=payload.get("actor_email"),
                actor_role=payload.get("actor_role"),
                details={
                    "stage": stage,
                    "status": "done",
                    "original_rows": prep.original_rows,
                    "clean_rows": prep.clean_rows,
                    "quarantine_count": prep.quarantine_count,
                    "clean_blob_path": prep.clean_blob_path,
                },
            )
            return _next(payload, stage=stage, clean_blob_path=prep.clean_blob_path)

        if not await _blob_exists(container.connection_string, container.container_name, file.blob_path):
            file.ingest_status = IngestStatus.NOT_INGESTED.value
            await db.commit()
            raise RuntimeError(f"source blob missing in Azure: {file.blob_path}")

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="skipped",
            reason="already_preprocessed_or_not_supported",
            file_id=file_id,
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage",
            level="info",
            trace_id=file_id,
            file_id=file_id,
            file_name=file.name,
            duration_ms=dur,
            actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"),
            actor_role=payload.get("actor_role"),
            details={"stage": stage, "status": "skipped", "reason": "already_preprocessed_or_not_supported"},
        )
        return _next(payload, stage=stage, clean_blob_path=file.blob_path)


async def parquet_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.PARQUET.value

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        blob_path = file.blob_path
        if not blob_path:
            raise RuntimeError("blob_path missing before parquet stage")

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)

        analytics = (
            await db.execute(select(FileAnalytics).where(FileAnalytics.file_id == file_id))
        ).scalar_one_or_none()
        if analytics and analytics.parquet_blob_path:
            ingest_logger.info(
                "ingest_stage",
                stage=stage,
                status="skipped",
                reason="parquet_already_exists",
                file_id=file_id,
            )
            return _next(payload, stage=stage, parquet_blob_path=analytics.parquet_blob_path)

        if not is_parquet_source_file(file.blob_path or file.name):
            ingest_logger.info(
                "ingest_stage",
                stage=stage,
                status="skipped",
                reason="not_csv_like",
                file_id=file_id,
            )
            return _next(payload, stage=stage, parquet_blob_path=None)

    await trigger_parquet_conversion(
        file_id=file_id,
        blob_path=blob_path,
        connection_string=container.connection_string,
        container_name=container.container_name,
    )

    async with _async_session() as db:
        analytics = (
            await db.execute(select(FileAnalytics).where(FileAnalytics.file_id == file_id))
        ).scalar_one_or_none()
        parquet_blob_path = analytics.parquet_blob_path if analytics else None

    dur = _ms(start)
    ingest_logger.info(
        "ingest_stage",
        stage=stage,
        status="done",
        file_id=file_id,
        parquet_blob_path=parquet_blob_path,
        duration_ms=dur,
    )
    await _db_log(
        event="ingest_stage",
        level="info",
        trace_id=file_id,
        file_id=file_id,
        duration_ms=dur,
        actor_user_id=payload.get("actor_user_id"),
        actor_email=payload.get("actor_email"),
        actor_role=payload.get("actor_role"),
        details={"stage": stage, "status": "done", "parquet_blob_path": parquet_blob_path},
    )
    return _next(payload, stage=stage, parquet_blob_path=parquet_blob_path)


async def metadata_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.METADATA.value

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)

        sample = await sample_file(
            blob_path=file.blob_path,
            connection_string=container.connection_string,
            container_name=container.container_name,
        )

        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            metadata = FileMetadata(id=str(uuid.uuid4()), file_id=file_id)
            db.add(metadata)

        metadata.blob_path = file.blob_path
        metadata.container_id = file.container_id
        metadata.columns_info = sample["columns_info"]
        metadata.row_count = sample["row_count"]
        metadata.sample_rows = sample["sample_rows"]
        metadata.ingest_error = None
        metadata.ingested_at = datetime.now(timezone.utc)
        await db.commit()

        column_types = {
            str(col.get("name")): col.get("type")
            for col in sample["columns_info"]
            if isinstance(col, dict) and col.get("name")
        }
        ingest_logger.info(
            "metadata_schema_detected",
            stage=stage,
            status="done",
            file_id=file_id,
            filename=file.name,
            row_count=sample["row_count"],
            column_types=column_types,
        )

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="done",
            file_id=file_id,
            columns=len(sample["columns_info"]),
            row_count=sample["row_count"],
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage",
            level="info",
            trace_id=file_id,
            file_id=file_id,
            file_name=file.name,
            duration_ms=dur,
            actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"),
            actor_role=payload.get("actor_role"),
            details={
                "stage": stage,
                "status": "done",
                "row_count": sample["row_count"],
                "columns": len(sample["columns_info"]),
            },
        )
        return _next(payload, stage=stage)


async def ai_description_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.AI_DESCRIPTION.value

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            raise RuntimeError("metadata missing before AI description stage")

        domain_tag: str | None = None
        column_glossary: dict[str, str] = {}
        if file.folder_id:
            folder = await db.get(Folder, file.folder_id)
            if folder:
                domain_tag = folder.domain_tag
                if not _is_schema_file(file.name):
                    column_glossary = await _load_schema_glossary(
                        folder_id=file.folder_id,
                        db=db,
                        connection_string=container.connection_string,
                        container_name=container.container_name,
                    )

        _org_ai = await _resolve_org_ai(db, container)

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        description = await generate_file_description(
            columns_info=metadata.columns_info or [],
            sample_rows=metadata.sample_rows or [],
            filename=file.name,
            domain_tag=domain_tag,
            column_glossary=column_glossary or None,
            org_ai=_org_ai,
        )

        # ── Escalation hook (gated; default OFF => no-op) ─────────────────────
        # generate_file_description returns the empty-fallback shape (summary ==
        # filename, no good_for/key_metrics) when the mini call's JSON parse
        # failed after its inner retry. Retry ONCE at the high tier.
        escalation_enabled, _ = _escalation_policy()
        if escalation_enabled:
            mini_failed = (
                description.get("summary", "") == file.name
                and not description.get("good_for")
                and not description.get("key_metrics")
            )
            if mini_failed:
                ingest_logger.info(
                    "ingest_stage_escalation", stage=stage, file_id=file_id, reason="json_parse_failed"
                )
                description = await generate_file_description(
                    columns_info=metadata.columns_info or [],
                    sample_rows=metadata.sample_rows or [],
                    filename=file.name,
                    domain_tag=domain_tag,
                    column_glossary=column_glossary or None,
                    tier="high",
                    org_ai=_org_ai,
                )

        metadata.ai_description = description.get("summary", "")
        metadata.good_for = description.get("good_for", [])
        metadata.key_metrics = description.get("key_metrics", [])
        metadata.key_dimensions = description.get("key_dimensions", [])

        if description.get("date_range_start"):
            try:
                metadata.date_range_start = date.fromisoformat(description["date_range_start"])
            except (ValueError, TypeError):
                pass
        if description.get("date_range_end"):
            try:
                metadata.date_range_end = date.fromisoformat(description["date_range_end"])
            except (ValueError, TypeError):
                pass

        await db.commit()

        file.ingest_status = IngestStatus.INGESTED.value
        metadata.ingest_error = None
        await db.commit()

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="done",
            file_id=file_id,
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
            file_name=file.name, duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
            domain_tag=domain_tag,
            details={"stage": stage, "status": "done", "summary_length": len(metadata.ai_description or "")},
        )

        try:
            from app.agent.catalog_cache import invalidate_catalog_cache  # noqa: PLC0415

            invalidate_catalog_cache()
        except Exception as exc:
            ingest_logger.warning("catalog_invalidate_failed", file_id=file_id, error=str(exc)[:200])

        return _next(payload, stage=stage)


async def ontology_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.ONTOLOGY.value

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            raise RuntimeError("metadata missing before ontology stage")

        column_glossary: dict[str, str] = {}
        if file.folder_id and not _is_schema_file(file.name):
            column_glossary = await _load_schema_glossary(
                folder_id=file.folder_id,
                db=db,
                connection_string=container.connection_string,
                container_name=container.container_name,
            )

        _org_ai = await _resolve_org_ai(db, container)

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)

        escalation_enabled, _ = _escalation_policy()
        if escalation_enabled:
            from app.services.column_role_resolver import (  # noqa: PLC0415
                resolve_column_roles_with_signal,
            )

            col_roles, role_src, role_evidence, parse_failed = await resolve_column_roles_with_signal(
                columns_info=metadata.columns_info or [],
                filename=file.name,
                glossary=column_glossary or None,
                semantic_config=container.semantic_config or None,
                org_ai=_org_ai,
            )
            # Escalation hook: mini call failed JSON parse after inner retry —
            # retry ONCE at the high tier.
            if parse_failed:
                ingest_logger.info(
                    "ingest_stage_escalation", stage=stage, file_id=file_id, reason="json_parse_failed"
                )
                col_roles, role_src, role_evidence, _ = await resolve_column_roles_with_signal(
                    columns_info=metadata.columns_info or [],
                    filename=file.name,
                    glossary=column_glossary or None,
                    semantic_config=container.semantic_config or None,
                    tier="high",
                    org_ai=_org_ai,
                )
        else:
            from app.services.column_role_resolver import resolve_column_roles  # noqa: PLC0415

            col_roles, role_src, role_evidence = await resolve_column_roles(
                columns_info=metadata.columns_info or [],
                filename=file.name,
                glossary=column_glossary or None,
                semantic_config=container.semantic_config or None,
                org_ai=_org_ai,
            )
        metadata.column_semantic_roles = col_roles or None
        metadata.role_source = role_src
        metadata.column_role_evidence = role_evidence or None
        await db.commit()

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="done",
            file_id=file_id,
            resolved=len(col_roles),
            source=role_src,
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
            file_name=file.name, duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
            details={"stage": stage, "status": "done", "resolved": len(col_roles), "source": role_src},
        )
        return _next(payload, stage=stage)


async def erp_classification_stage(payload: Payload) -> Payload:
    """Classify a file's business context (source system / module / polarity /
    process role) from evidence already produced upstream. Data-driven (LLM +
    semantic-role corroboration), never hardcoded. Non-fatal: any failure or
    low confidence persists an 'unknown' row and the pipeline continues."""
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.ERP_CLASSIFICATION.value

    # Feature flag — default ON, but never blocks ingestion if disabled.
    from app.core.config import get_settings as _get_settings  # noqa: PLC0415
    if not bool(getattr(_get_settings(), "ERP_CLASSIFICATION_ENABLED", True)):
        return _next(payload, stage=stage)

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            raise RuntimeError("metadata missing before erp_classification stage")

        _org_ai = await _resolve_org_ai(db, container)

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)

        from app.services.erp.classifier import classify_file, schema_fingerprint  # noqa: PLC0415
        from app.models.erp_classification import ErpClassification as ErpRow  # noqa: PLC0415

        _container_id = metadata.container_id or container.id
        fingerprint = schema_fingerprint(metadata.columns_info or [])

        # ── Schema-fingerprint cache ──────────────────────────────────────────
        # If a sibling file in this container with the SAME column shape was
        # already reliably classified, reuse it and skip the LLM call entirely.
        # This is the big speed lever for ERP archives full of identical tables.
        clf = None
        if fingerprint:
            sibling = (await db.execute(
                select(ErpRow).where(
                    ErpRow.container_id == _container_id,
                    ErpRow.schema_fingerprint == fingerprint,
                    ErpRow.file_id != file_id,
                    ErpRow.source.in_(("llm", "human_override")),
                ).limit(1)
            )).scalar_one_or_none()
            if sibling is not None:
                from app.services.erp.classifier import ErpClassification as _Clf  # noqa: PLC0415

                clf = _Clf(
                    source_system=sibling.source_system,
                    erp_module=sibling.erp_module,
                    domain_polarity=sibling.domain_polarity,  # type: ignore[arg-type]
                    process_role=sibling.process_role,
                    grain=sibling.grain or "",
                    confidence=sibling.confidence,
                    evidence=[f"cache: reused identical schema from {sibling.file_id}"],
                    source="cache",
                    model_version=sibling.model_version or "",
                )
                ingest_logger.info("erp_classification_cache_hit", file_id=file_id, fingerprint=fingerprint[:12])

        if clf is None:
            clf = await classify_file(
                filename=file.name,
                columns_info=metadata.columns_info or [],
                sample_rows=metadata.sample_rows or [],
                ai_description=metadata.ai_description,
                column_semantic_roles=metadata.column_semantic_roles or {},
                org_ai=_org_ai,
            )

            # ── Escalation hook (gated; default OFF => no-op) ─────────────────
            # Re-run ONCE at the high tier when the mini classification is below
            # the confidence threshold OR produced no usable result (invalid
            # JSON after its inner retry => source != "llm").
            escalation_enabled, threshold = _escalation_policy()
            if escalation_enabled:
                low_confidence = clf.confidence < threshold
                mini_failed = clf.source != "llm"
                if low_confidence or mini_failed:
                    ingest_logger.info(
                        "ingest_stage_escalation",
                        stage=stage,
                        file_id=file_id,
                        reason="low_confidence" if low_confidence else "json_parse_failed",
                        confidence=clf.confidence,
                    )
                    escalated = await classify_file(
                        filename=file.name,
                        columns_info=metadata.columns_info or [],
                        sample_rows=metadata.sample_rows or [],
                        ai_description=metadata.ai_description,
                        column_semantic_roles=metadata.column_semantic_roles or {},
                        tier="high",
                        org_ai=_org_ai,
                    )
                    # Keep the better result; never regress to a less-reliable one.
                    if escalated.is_reliable or escalated.confidence >= clf.confidence:
                        clf = escalated

        # Upsert one row per file (re-ingest replaces it).
        existing = (
            await db.execute(select(ErpRow).where(ErpRow.file_id == file_id))
        ).scalar_one_or_none()
        if existing:
            existing.container_id = _container_id
            existing.source_system = clf.source_system
            existing.erp_module = clf.erp_module
            existing.domain_polarity = clf.domain_polarity
            existing.process_role = clf.process_role
            existing.grain = clf.grain or None
            existing.confidence = clf.confidence
            existing.evidence = clf.evidence or []
            existing.source = clf.source
            existing.model_version = clf.model_version or None
            existing.schema_fingerprint = fingerprint or None
            existing.computed_at = datetime.now(timezone.utc)
        else:
            db.add(ErpRow(
                container_id=_container_id,
                file_id=file_id,
                source_system=clf.source_system,
                erp_module=clf.erp_module,
                domain_polarity=clf.domain_polarity,
                process_role=clf.process_role,
                grain=clf.grain or None,
                confidence=clf.confidence,
                evidence=clf.evidence or [],
                source=clf.source,
                model_version=clf.model_version or None,
                schema_fingerprint=fingerprint or None,
            ))
        await db.commit()

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage", stage=stage, status="done", file_id=file_id,
            source_system=clf.source_system, polarity=clf.domain_polarity,
            confidence=clf.confidence, reliable=clf.is_reliable, duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
            file_name=file.name, duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
            details={
                "stage": stage, "status": "done", "source_system": clf.source_system,
                "module": clf.erp_module, "polarity": clf.domain_polarity,
                "process_role": clf.process_role, "confidence": clf.confidence,
            },
        )
        return _next(payload, stage=stage)


async def embedding_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.EMBEDDING.value

    async with _async_session() as db:
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            raise RuntimeError("metadata missing before embedding stage")

        _org_ai = None
        if metadata.container_id:
            container = await db.get(ContainerConfig, metadata.container_id)
            if container:
                _org_ai = await _resolve_org_ai(db, container)

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        search_text = build_search_text(metadata)
        metadata.search_text = search_text
        metadata.description_embedding = await embed_text(search_text, org_ai=_org_ai)
        await db.commit()

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="done",
            file_id=file_id,
            search_text_len=len(search_text),
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
            duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
            details={"stage": stage, "status": "done", "search_text_len": len(search_text)},
        )
        return _next(payload, stage=stage)


async def opensearch_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.OPENSEARCH.value

    async with _async_session() as db:
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            raise RuntimeError("metadata missing before OpenSearch stage")

        from app.retrieval.opensearch_indexer import index_metadata_document  # noqa: PLC0415

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        await index_metadata_document(metadata, db)

    dur = _ms(start)
    ingest_logger.info(
        "ingest_stage",
        stage=stage,
        status="done",
        file_id=file_id,
        duration_ms=dur,
    )
    await _db_log(
        event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
        duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
        actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
        details={"stage": stage, "status": "done"},
    )
    return _next(payload, stage=stage)


async def analytics_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.ANALYTICS.value

    async with _async_session() as db:
        file, container = await _load_file_and_container(db, file_id)
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not metadata:
            raise RuntimeError("metadata missing before analytics stage")

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        analytics = await compute_and_store_analytics(
            file_id=file_id,
            blob_path=file.blob_path,
            connection_string=container.connection_string,
            container_name=container.container_name,
            columns_info=metadata.columns_info or [],
            db=db,
        )

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="done",
            file_id=file_id,
            row_count=analytics.row_count,
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
            duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
            details={"stage": stage, "status": "done", "row_count": analytics.row_count},
        )
        return _next(payload, stage=stage)


async def relationship_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.RELATIONSHIPS.value

    async with _async_session() as db:
        file = await db.get(File, file_id)
        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if not file or not metadata:
            raise RuntimeError("file or metadata missing before relationship stage")

        from app.models.column_key_registry import ColumnKeyRegistry  # noqa: PLC0415
        from app.models.file_relationship import FileRelationship  # noqa: PLC0415
        from app.models.semantic_layer import SemanticRelationship  # noqa: PLC0415
        from app.services.relationship_index import is_dictionary_like_path  # noqa: PLC0415

        if is_dictionary_like_path(file.name):
            await db.execute(delete(SemanticRelationship).where(
                (SemanticRelationship.file_a_id == file_id) | (SemanticRelationship.file_b_id == file_id)
            ))
            await db.execute(delete(FileRelationship).where(
                (FileRelationship.file_a_id == file_id) | (FileRelationship.file_b_id == file_id)
            ))
            await db.execute(delete(ColumnKeyRegistry).where(ColumnKeyRegistry.file_id == file_id))
            await db.commit()

            dur = _ms(start)
            ingest_logger.info(
                "ingest_stage",
                stage=stage,
                status="skipped",
                reason="dictionary_file_not_joinable",
                file_id=file_id,
                duration_ms=dur,
            )
            await _db_log(
                event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
                duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
                actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
                details={"stage": stage, "status": "skipped", "reason": "dictionary_file_not_joinable"},
            )
            return _next(payload, stage=stage, relationships_created=0)

        from app.services.relationship_detector import detect_relationships  # noqa: PLC0415

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        relationship_count = await detect_relationships(
            file_id=file_id,
            blob_path=file.blob_path or metadata.blob_path,
            columns_info=metadata.columns_info or [],
            db=db,
        )

        dur = _ms(start)
        ingest_logger.info(
            "ingest_stage",
            stage=stage,
            status="done",
            file_id=file_id,
            relationships_created=relationship_count,
            duration_ms=dur,
        )
        await _db_log(
            event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
            duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
            actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
            details={"stage": stage, "status": "done", "relationships_created": relationship_count},
        )
        return _next(payload, stage=stage, relationships_created=relationship_count)


async def semantic_layer_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.SEMANTIC_LAYER.value

    async with _async_session() as db:
        from app.services.semantic_layer_builder import build_semantic_layer_for_file  # noqa: PLC0415

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        result = await build_semantic_layer_for_file(file_id, db)

    dur = _ms(start)
    ingest_logger.info(
        "ingest_stage",
        stage=stage,
        status="done",
        file_id=file_id,
        entity=result.get("entity"),
        relationships=result.get("relationships"),
        duration_ms=dur,
    )
    await _db_log(
        event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
        duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
        actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
        details={
            "stage": stage, "status": "done",
            "entity": result.get("entity"),
            "relationships": result.get("relationships"),
        },
    )
    return _next(payload, stage=stage, semantic_layer=result)


async def semantic_enrichment_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    start = time.perf_counter()
    stage = StageName.SEMANTIC_ENRICHMENT.value

    async with _async_session() as db:
        from app.services.semantic_enrichment import run_semantic_enrichment_for_file  # noqa: PLC0415

        ingest_logger.info("ingest_stage", stage=stage, status="started", file_id=file_id)
        result = await run_semantic_enrichment_for_file(file_id, db)

    dur = _ms(start)
    ingest_logger.info(
        "ingest_stage",
        stage=stage,
        status="done",
        file_id=file_id,
        additions=result.get("additions", 0),
        skipped=result.get("skipped", False),
        duration_ms=dur,
    )
    await _db_log(
        event="ingest_stage", level="info", trace_id=file_id, file_id=file_id,
        duration_ms=dur, actor_user_id=payload.get("actor_user_id"),
        actor_email=payload.get("actor_email"), actor_role=payload.get("actor_role"),
        details={
            "stage": stage,
            "status": "done",
            **{
                k: result.get(k)
                for k in ("additions", "skipped", "reason", "neighbors_used", "role_groups_used")
                if k in result
            },
        },
    )
    return _next(payload, stage=stage, semantic_enrichment=result)


async def complete_ingestion_stage(payload: Payload) -> Payload:
    file_id = payload["file_id"]
    stage = StageName.COMPLETE.value

    async with _async_session() as db:
        file = await db.get(File, file_id)
        if file:
            file.ingest_status = IngestStatus.INGESTED.value
            await db.commit()

        # ── Compute and persist ingestion confidence score ────────────────────────
        # Runs after all stages so both role evidence and relationships are present.
        try:
            from app.models.file_relationship import FileRelationship  # noqa: PLC0415
            from app.services.ingestion_confidence import compute_ingestion_confidence  # noqa: PLC0415

            meta = (
                await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
            ).scalar_one_or_none()
            rels = (
                await db.execute(
                    select(FileRelationship).where(
                        (FileRelationship.file_a_id == file_id)
                        | (FileRelationship.file_b_id == file_id)
                    )
                )
            ).scalars().all()

            if meta:
                ing_conf = compute_ingestion_confidence(meta, list(rels))
                meta.ingestion_confidence_score = ing_conf.overall
                meta.ingestion_confidence_signals = ing_conf.signals
                await db.commit()
                ingest_logger.info(
                    "ingestion_confidence",
                    file_id=file_id,
                    overall=ing_conf.overall,
                    level=ing_conf.level,
                    signals=ing_conf.signals,
                )
        except Exception as exc:
            ingest_logger.warning(
                "ingestion_confidence_failed", file_id=file_id, error=str(exc)[:200]
            )

    try:
        from app.agent.catalog_cache import invalidate_catalog_cache  # noqa: PLC0415

        invalidate_catalog_cache()
    except Exception as exc:
        ingest_logger.warning("catalog_invalidate_failed", file_id=file_id, error=str(exc)[:200])

    # ── Recompile the Danta Semantic Contract for this container ──────────────
    # Hash-guarded: writes only when the inputs actually changed, so per-file
    # recompiles are cheap. Non-fatal — a contract failure never fails ingestion.
    try:
        from app.core.config import get_settings as _get_settings  # noqa: PLC0415
        if bool(getattr(_get_settings(), "SEMANTIC_CONTRACT_ENABLED", True)):
            from app.services.contract.builder import compile_and_store_contract  # noqa: PLC0415

            _cid = None
            async with _async_session() as _cdb:
                _f = await _cdb.get(File, file_id)
                _cid = _f.container_id if _f else None
                if _cid:
                    await compile_and_store_contract(_cdb, _cid)
    except Exception as exc:
        ingest_logger.warning("contract_compile_failed", file_id=file_id, error=str(exc)[:200])

    ingest_logger.info("ingest_stage", stage=stage, status="done", file_id=file_id)
    await _db_log(
        event="chain_end",
        level="info",
        trace_id=file_id,
        file_id=file_id,
        actor_user_id=payload.get("actor_user_id"),
        actor_email=payload.get("actor_email"),
        actor_role=payload.get("actor_role"),
        details={"stage": stage, "status": "done"},
    )
    return _next(payload, stage=stage, status=PayloadStatus.DONE.value)


async def mark_ingestion_failed(file_id: str, stage: str, exc: BaseException) -> None:
    error = f"Ingestion stage {stage} failed: {exc}"[:1000]
    async with _async_session() as db:
        file = await db.get(File, file_id)
        if file:
            file.ingest_status = IngestStatus.FAILED.value

        metadata = (
            await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
        ).scalar_one_or_none()
        if metadata:
            metadata.ingest_error = error
        elif file:
            db.add(FileMetadata(
                id=str(uuid.uuid4()),
                file_id=file_id,
                blob_path=file.blob_path,
                container_id=file.container_id,
                ingest_error=error,
            ))

        await db.commit()

    await _db_log(
        event="ingest_failed",
        level="error",
        trace_id=file_id,
        file_id=file_id,
        details={"stage": stage, "error": error},
    )
