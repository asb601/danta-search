import asyncio
import time
import uuid
from datetime import date
from pathlib import Path

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.ai_client import generate_file_description
from app.core.config import get_settings
from app.core.database import async_session as _async_session
from app.core.duckdb_client import sample_file
from app.core.logger import ingest_logger
from app.retrieval.embeddings import build_search_text, embed_text
from app.models.container import ContainerConfig
from app.models.file import File
from app.models.file_analytics import FileAnalytics
from app.models.file_metadata import FileMetadata
from app.models.folder import Folder
from app.services.analytics_service import compute_and_store_analytics, trigger_parquet_conversion
from app.services.data_preprocessor import preprocess_file, probe_raw_csv
from app.services.ingestion_config import (
    IngestStatus,
    glossary_filename_tokens,
    is_excel_ingest_file,
    parquet_extension,
    preprocess_extensions,
)

_PREPROCESS_EXTS = preprocess_extensions(dotted=True)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _is_schema_file(filename: str) -> bool:
    """Return True if the filename signals a column glossary / schema file."""
    stem = Path(filename).stem.lower()
    return any(token in stem for token in glossary_filename_tokens())


async def _load_schema_glossary(
    folder_id: str,
    db: AsyncSession,
    connection_string: str,
    container_name: str,
) -> dict[str, str]:
    """Find a schema file in the same folder and parse it as a column glossary.

    Schema file format (auto-detected):
      - First column = raw column code (e.g. WRBTR, BUKRS)
      - Second column = business meaning (e.g. Amount in Local Currency)
      Optional third column = longer description (appended to the meaning)

    Returns an empty dict if no schema file is found or parsing fails.
    """
    # Find schema files in this folder (already uploaded, any ingest status)
    result = await db.execute(
        select(File).where(
            File.folder_id == folder_id,
            File.blob_path.isnot(None),
        )
    )
    siblings = result.scalars().all()

    schema_file: File | None = None
    for sib in siblings:
        if _is_schema_file(sib.name):
            schema_file = sib
            break

    if not schema_file or not schema_file.blob_path:
        return {}

    ingest_logger.info(
        "schema_file_found",
        schema_file=schema_file.name,
        blob_path=schema_file.blob_path,
        folder_id=folder_id,
    )

    def _parse(blob_path: str) -> dict[str, str]:
        """Read schema file synchronously with DuckDB, return code->meaning dict."""
        try:
            import duckdb  # noqa: PLC0415
            import os  # noqa: PLC0415

            _CA = "/etc/ssl/certs/ca-certificates.crt"
            if os.path.exists(_CA):
                os.environ.setdefault("CURL_CA_BUNDLE", _CA)

            conn = duckdb.connect()
            try:
                conn.execute("INSTALL azure IF NOT EXISTS;")
            except Exception:
                try:
                    conn.execute("INSTALL azure;")
                except Exception:
                    pass
            conn.execute("LOAD azure;")
            conn.execute("SET azure_transport_option_type = 'curl';")
            safe_cs = connection_string.replace("'", "''")
            conn.execute(f"SET azure_storage_connection_string='{safe_cs}';")

            sample_rows = max(1, int(get_settings().INGEST_DUCKDB_SAMPLE_ROWS))
            url = f"azure://{container_name}/{blob_path}".replace("'", "''")
            rows = conn.execute(
                f"SELECT * FROM read_csv_auto('{url}', header=true, "
                f"ignore_errors=true, sample_size={sample_rows}) LIMIT {sample_rows}"
            ).fetchall()
            cols = [d[0] for d in conn.description]

            if len(cols) < 2:
                return {}

            glossary: dict[str, str] = {}
            for row in rows:
                code = str(row[0]).strip() if row[0] is not None else ""
                meaning = str(row[1]).strip() if row[1] is not None else ""
                if len(cols) >= 3 and row[2]:
                    extra = str(row[2]).strip()
                    if extra:
                        meaning = f"{meaning} ({extra})"
                if code and meaning:
                    glossary[code] = meaning
            return glossary
        except Exception as exc:
            ingest_logger.warning(
                "schema_parse_failed",
                blob_path=blob_path,
                error=str(exc)[:300],
            )
            return {}

    glossary = await asyncio.to_thread(_parse, schema_file.blob_path)
    ingest_logger.info(
        "schema_glossary_loaded",
        entry_count=len(glossary),
        sample=dict(list(glossary.items())[:max(0, int(get_settings().INGEST_LOG_SAMPLE_ITEMS))]),
    )
    return glossary


def _ensure_trace(file_id: str) -> None:
    """Bind a trace_id if one isn't already set (background tasks from sync/upload)."""
    ctx = structlog.contextvars.get_contextvars()
    if "trace_id" not in ctx:
        structlog.contextvars.bind_contextvars(
            trace_id=f"ingest-{uuid.uuid4().hex[:12]}",
            pipeline="ingest",
            file_id=file_id,
        )


async def _delete_blob_silent(connection_string: str, container_name: str, blob_path: str) -> None:
    """Delete a blob from Azure. Swallows all errors (blob may not exist)."""
    def _run() -> None:
        try:
            from azure.storage.blob import BlobServiceClient  # noqa: PLC0415
            BlobServiceClient.from_connection_string(connection_string) \
                .get_blob_client(container=container_name, blob=blob_path) \
                .delete_blob()
        except Exception:
            pass  # blob may not exist — that's fine
    await asyncio.to_thread(_run)


async def ingest_file(file_id: str, db: AsyncSession) -> None:
    """
    Sample a file with DuckDB, generate AI description, embed, and kick off Parquet.
    Updates file.ingest_status throughout: pending → ingested | failed.

    Parallelism strategy for CSV/text files:
      Preprocessing (full file clean + re-upload) is fired as a background task
      immediately, while DuckDB samples the raw file concurrently.  Once the
      sample is in hand the AI and embedding steps run — all before preprocessing
      finishes.  This cuts perceived ingest time from O(file_size) to ~30 s for
      any size CSV.  Preprocessing is awaited only before Parquet conversion so
      that conversion uses the clean CSV.

    For Excel files preprocessing must complete before DuckDB can sample
    (DuckDB cannot read .xlsx natively), so they remain sequential.
    """
    _ensure_trace(file_id)
    pipeline_start = time.perf_counter()

    # Tracked so we can cancel on error
    preprocess_task: "asyncio.Task | None" = None
    # Holds the latest PreprocessResult regardless of which code path ran it
    _prep_result = None
    log_sample_items = max(0, int(get_settings().INGEST_LOG_SAMPLE_ITEMS))

    try:
        file = await db.get(File, file_id)
        if not file or not file.blob_path:
            ingest_logger.warning("chain_skip", reason="file or blob_path missing")
            return

        container = await db.get(ContainerConfig, file.container_id)
        if not container:
            ingest_logger.warning("chain_skip", reason="container not found")
            return

        ingest_logger.info("chain_start", filename=file.name, blob_path=file.blob_path,
                           container=container.container_name)

        file.ingest_status = IngestStatus.PENDING.value
        await db.commit()

        ext = Path(file.name).suffix.lower()
        raw_blob_path = file.blob_path
        is_excel = is_excel_ingest_file(file.name)
        already_preprocessed = bool(file.is_preprocessed)

        # ── Pre-flight: clear stale parquet path on retry/reingest ───────────
        # If a previous parquet was generated, drop the path so a fresh parquet
        # is produced from the (possibly newly cleaned) CSV.  The actual blob
        # is overwritten safely later by parquet_service (overwrite=True).
        if ext in _PREPROCESS_EXTS and not already_preprocessed:
            analytics_row = (
                await db.execute(select(FileAnalytics).where(FileAnalytics.file_id == file_id))
            ).scalar_one_or_none()
            if analytics_row and analytics_row.parquet_blob_path:
                analytics_row.parquet_blob_path = None
                analytics_row.parquet_size_bytes = None
                await db.commit()
                ingest_logger.info("cleanup", action="cleared_parquet_path", file_id=file_id)

        # ── Step 0 · Preprocess ───────────────────────────────────────────────
        # Skipped entirely when file.is_preprocessed=True — the clean CSV
        # (blob_path) and Parquet already exist from a previous ingestion run.
        if ext in _PREPROCESS_EXTS and not already_preprocessed:
            if is_excel:
                # Excel: DuckDB cannot read .xlsx — preprocessing must finish first.
                step_start = time.perf_counter()
                ingest_logger.info("step", step="0/6", name="preprocess", status="started",
                                   blob_path=raw_blob_path, ext=ext, mode="sequential")
                try:
                    prep = await preprocess_file(
                        blob_path=raw_blob_path, file_name=file.name, file_id=file_id,
                        connection_string=container.connection_string,
                        container_name=container.container_name,
                        cleaning_config=container.cleaning_config,
                    )
                    _prep_result = prep
                    file.blob_path = prep.clean_blob_path
                    file.is_preprocessed = True
                    await db.commit()
                    ingest_logger.info("step", step="0/6", name="preprocess", status="done",
                                       clean_blob_path=prep.clean_blob_path,
                                       original_rows=prep.original_rows,
                                       clean_rows=prep.clean_rows,
                                       duration_ms=_ms(step_start))
                except Exception as prep_exc:
                    raise RuntimeError(
                        f"Excel preprocessing failed — cannot ingest: {prep_exc}"
                    ) from prep_exc
            else:
                # CSV/text: read the first 256 KB to decide whether DuckDB can
                # sample the raw file reliably:
                #   • Non-UTF-8 encoding → DuckDB garbles strings
                #   • Leading junk rows  → DuckDB uses them as column names,
                #                          making the AI description wrong
                # For everything else (dirty nulls like "N/A", whitespace,
                # control chars in values), DuckDB handles it fine with
                # ignore_errors=true and the AI still gets an accurate description.
                probe_start = time.perf_counter()
                probe = await probe_raw_csv(
                    blob_path=raw_blob_path, file_name=file.name,
                    connection_string=container.connection_string,
                    container_name=container.container_name,
                )
                ingest_logger.info("step", step="0/6", name="probe",
                                   safe_for_raw_sample=probe.safe_for_raw_sample,
                                   encoding=probe.encoding,
                                   header_row_idx=probe.header_row_idx,
                                   reason=probe.reason or "ok",
                                   duration_ms=_ms(probe_start))

                if probe.safe_for_raw_sample:
                    # Fire preprocessing in background; sample raw file immediately.
                    ingest_logger.info("step", step="0/6", name="preprocess",
                                       status="started_async", mode="parallel")
                    preprocess_task = asyncio.create_task(
                        preprocess_file(
                            blob_path=raw_blob_path, file_name=file.name, file_id=file_id,
                            connection_string=container.connection_string,
                            container_name=container.container_name,
                            cleaning_config=container.cleaning_config,
                        )
                    )
                else:
                    # Unsafe to sample raw — wait for preprocessing to finish first.
                    ingest_logger.info("step", step="0/6", name="preprocess",
                                       status="started", mode="sequential",
                                       reason=probe.reason)
                    step_start = time.perf_counter()
                    try:
                        prep = await preprocess_file(
                            blob_path=raw_blob_path, file_name=file.name, file_id=file_id,
                            connection_string=container.connection_string,
                            container_name=container.container_name,
                            cleaning_config=container.cleaning_config,
                        )
                        _prep_result = prep
                        file.blob_path = prep.clean_blob_path
                        file.is_preprocessed = True
                        await db.commit()
                        ingest_logger.info("step", step="0/6", name="preprocess",
                                           status="done",
                                           clean_blob_path=prep.clean_blob_path,
                                           original_rows=prep.original_rows,
                                           clean_rows=prep.clean_rows,
                                           duration_ms=_ms(step_start))
                    except Exception as prep_exc:
                        ingest_logger.warning("step", step="0/6", name="preprocess",
                                              status="skipped",
                                              error=str(prep_exc)[:400],
                                              duration_ms=_ms(step_start))
        else:
            if already_preprocessed:
                ingest_logger.info("step", step="0/6", name="preprocess",
                                   status="skipped", reason="already_preprocessed",
                                   blob_path=file.blob_path)

        # ── Guard: verify the source blob still exists in Azure ──────────────
        # With in-place overwrite, blob_path always points to the single source
        # of truth.  If it's missing, the user deleted it from the container —
        # mark not_ingested so the next sync re-discovers it cleanly.
        if file.blob_path:
            _conn_str = container.connection_string
            _cont_name = container.container_name

            def _check_blob(path: str) -> bool:
                try:
                    from azure.storage.blob import BlobServiceClient  # noqa: PLC0415
                    bc = BlobServiceClient.from_connection_string(_conn_str)
                    return bc.get_blob_client(container=_cont_name, blob=path).exists()
                except Exception:
                    return False

            if not await asyncio.to_thread(_check_blob, file.blob_path):
                ingest_logger.warning(
                    "blob_missing_in_azure",
                    blob_path=file.blob_path,
                    action="resetting to not_ingested",
                )
                stale_meta = (
                    await db.execute(select(FileMetadata).where(FileMetadata.file_id == file_id))
                ).scalar_one_or_none()
                if stale_meta:
                    await db.delete(stale_meta)
                stale_analytics = (
                    await db.execute(select(FileAnalytics).where(FileAnalytics.file_id == file_id))
                ).scalar_one_or_none()
                if stale_analytics:
                    await db.delete(stale_analytics)
                file.is_preprocessed = False
                file.ingest_status = IngestStatus.NOT_INGESTED.value
                await db.commit()
                return

        # ── Step 1/6 · Sample with DuckDB ────────────────────────────────────
        # For CSV/text, samples the raw file while preprocessing runs in background.
        # Falls back to awaiting the clean CSV only if the raw file is unreadable.
        step_start = time.perf_counter()
        ingest_logger.info("step", step="1/6", name="duckdb_sample", status="started",
                           blob_path=file.blob_path)

        try:
            sample = await sample_file(
                blob_path=file.blob_path,
                connection_string=container.connection_string,
                container_name=container.container_name,
            )
        except Exception as raw_exc:
            if preprocess_task is None:
                raise
            # Raw file too dirty — wait for the clean CSV then retry
            ingest_logger.warning("step", step="1/6", name="duckdb_sample",
                                  status="raw_failed_awaiting_preprocess",
                                  error=str(raw_exc)[:200])
            step_p = time.perf_counter()
            try:
                prep = await preprocess_task
                preprocess_task = None
            except Exception as prep_exc:
                raise RuntimeError(
                    f"Both raw DuckDB sample and preprocessing failed: {prep_exc}"
                ) from prep_exc
            file.blob_path = prep.clean_blob_path
            await db.commit()
            ingest_logger.info("step", step="0/6", name="preprocess", status="done_fallback",
                               clean_blob_path=prep.clean_blob_path,
                               duration_ms=_ms(step_p))
            sample = await sample_file(
                blob_path=file.blob_path,
                connection_string=container.connection_string,
                container_name=container.container_name,
            )

        ingest_logger.info("step", step="1/6", name="duckdb_sample", status="done",
                           columns=len(sample["columns_info"]),
                           column_names=sample["column_names"],
                           row_count=sample["row_count"],
                           sample_row_count=len(sample["sample_rows"]),
                           duration_ms=_ms(step_start))

        # ── Step 2/6 · AI description ─────────────────────────────────────────
        step_start = time.perf_counter()
        ingest_logger.info("step", step="2/6", name="ai_description", status="started",
                           filename=file.name)

        # Resolve domain context from folder
        domain_tag: str | None = None
        column_glossary: dict[str, str] = {}
        if file.folder_id:
            folder = await db.get(Folder, file.folder_id)
            if folder:
                domain_tag = folder.domain_tag
                # Load schema glossary from sibling schema file in same folder
                if not _is_schema_file(file.name):  # don't try to glossary a schema file itself
                    column_glossary = await _load_schema_glossary(
                        folder_id=file.folder_id,
                        db=db,
                        connection_string=container.connection_string,
                        container_name=container.container_name,
                    )

        description = await generate_file_description(
            columns_info=sample["columns_info"],
            sample_rows=sample["sample_rows"],
            filename=file.name,
            domain_tag=domain_tag,
            column_glossary=column_glossary or None,
        )

        ingest_logger.info("step", step="2/6", name="ai_description", status="done",
                           summary=description.get("summary", "")[:200],
                           good_for=description.get("good_for", []),
                           metrics=description.get("key_metrics", []),
                           dimensions=description.get("key_dimensions", []),
                           date_range=f"{description.get('date_range_start')} → {description.get('date_range_end')}",
                           duration_ms=_ms(step_start))

        # ── Step 3/6 · Save metadata ──────────────────────────────────────────
        step_start = time.perf_counter()
        ingest_logger.info("step", step="3/6", name="save_metadata", status="started")

        result = await db.execute(
            select(FileMetadata).where(FileMetadata.file_id == file_id)
        )
        metadata = result.scalar_one_or_none()
        is_new = metadata is None
        if not metadata:
            metadata = FileMetadata(id=str(uuid.uuid4()), file_id=file_id)
            db.add(metadata)

        # blob_path here may still be the raw path for CSV — updated below after
        # preprocessing finishes.  All query-time lookups use file.blob_path, not
        # metadata.blob_path, so this is safe.
        metadata.blob_path = file.blob_path
        metadata.container_id = file.container_id
        metadata.columns_info = sample["columns_info"]
        metadata.row_count = sample["row_count"]
        metadata.ai_description = description.get("summary", "")
        metadata.good_for = description.get("good_for", [])
        metadata.key_metrics = description.get("key_metrics", [])
        metadata.key_dimensions = description.get("key_dimensions", [])
        metadata.sample_rows = sample["sample_rows"]
        metadata.ingest_error = None

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
        ingest_logger.info("step", step="3/6", name="save_metadata", status="done",
                           action="created" if is_new else "updated",
                           duration_ms=_ms(step_start))

        # ── Step 3b/6 · Resolve column semantic roles (ontology layer) ────────
        # Paid once at ingest time. Stored forever. Zero LLM cost at query time.
        # Order: Tier 0 (glossary) → Tier 1 (heuristic) → Tier 2 (LLM fallback)
        step_start = time.perf_counter()
        ingest_logger.info("step", step="3b/6", name="resolve_roles", status="started")
        try:
            from app.services.column_role_resolver import resolve_column_roles  # noqa: PLC0415
            col_roles, role_src, role_evidence = await resolve_column_roles(
                columns_info=sample["columns_info"],
                filename=file.name,
                glossary=column_glossary or None,
            )
            metadata.column_semantic_roles = col_roles or None
            metadata.role_source = role_src
            metadata.column_role_evidence = role_evidence or None
            await db.commit()
            ingest_logger.info("step", step="3b/6", name="resolve_roles", status="done",
                               resolved=len(col_roles),
                               source=role_src,
                               duration_ms=_ms(step_start))
        except Exception as role_exc:
            # Non-fatal — file remains fully usable, relationships just won't be
            # role-indexed for this file until re-ingested with a better glossary.
            ingest_logger.warning("step", step="3b/6", name="resolve_roles", status="failed",
                                  error=str(role_exc)[:200],
                                  duration_ms=_ms(step_start))

        # ── Step 4/6 · Build search text + embed ─────────────────────────────
        step_start = time.perf_counter()
        ingest_logger.info("step", step="4/6", name="embed_metadata", status="started")

        try:
            search_text = build_search_text(metadata)
            metadata.search_text = search_text
            metadata.description_embedding = await embed_text(search_text)
            await db.commit()
            try:
                from app.retrieval.opensearch_indexer import index_metadata_document  # noqa: PLC0415
                await index_metadata_document(metadata, db)
            except Exception as os_exc:
                ingest_logger.warning(
                    "opensearch_index_step_failed",
                    file_id=file_id,
                    error=str(os_exc)[:300],
                )
            ingest_logger.info("step", step="4/6", name="embed_metadata", status="done",
                               search_text_len=len(search_text),
                               has_embedding=metadata.description_embedding is not None
                                             and any(x != 0.0 for x in (metadata.description_embedding or [])),
                               duration_ms=_ms(step_start))
        except Exception as embed_exc:
            # Embedding failure is non-fatal — file is searchable via BM25/trgm
            ingest_logger.warning("step", step="4/6", name="embed_metadata", status="failed",
                                  error=str(embed_exc)[:200],
                                  duration_ms=_ms(step_start))

        # ── Mark ingested — file is now AI-described and searchable ──────────
        # Preprocessing (if still running) finishes below before Parquet conversion.
        file.ingest_status = IngestStatus.INGESTED.value
        await db.commit()

        # ── Await background preprocessing to get the clean CSV path ─────────
        # Parquet conversion needs the clean CSV (normalised column names, types).
        clean_blob_path = file.blob_path
        if preprocess_task is not None:
            step_start = time.perf_counter()
            try:
                prep = await preprocess_task
                preprocess_task = None
                _prep_result = prep
                clean_blob_path = prep.clean_blob_path
                file.blob_path = clean_blob_path
                file.is_preprocessed = True
                metadata.blob_path = clean_blob_path
                await db.commit()
                ingest_logger.info("step", step="0/6", name="preprocess", status="done",
                                   clean_blob_path=clean_blob_path,
                                   original_rows=prep.original_rows,
                                   clean_rows=prep.clean_rows,
                                   rows_dropped=prep.rows_dropped,
                                   cols_renamed=len(prep.cols_renamed),
                                   warnings=prep.warnings[:log_sample_items],
                                   duration_ms=_ms(step_start))
            except Exception as prep_exc:
                # Non-fatal: Parquet conversion will use the raw CSV (DuckDB handles it)
                ingest_logger.warning("step", step="0/6", name="preprocess", status="failed",
                                      error=str(prep_exc)[:400],
                                      duration_ms=_ms(step_start))

        # ── Step 5/5 · Analytics + Parquet conversion ─────────────────────────
        step_start = time.perf_counter()
        ingest_logger.info("step", step="5/5", name="compute_analytics", status="started")

        try:
            async with _async_session() as analytics_db:
                analytics = await compute_and_store_analytics(
                    file_id=file_id,
                    blob_path=clean_blob_path,
                    connection_string=container.connection_string,
                    container_name=container.container_name,
                    columns_info=sample["columns_info"],
                    db=analytics_db,
                )
            ingest_logger.info("step", step="5/5", name="compute_analytics", status="done",
                               row_count=analytics.row_count,
                               duration_ms=_ms(step_start))
            # Persist cleaning audit data from the preprocessing step.
            # Stored on the FileAnalytics row so ops can audit dropped rows without
            # reprocessing the file.
            if _prep_result is not None:
                analytics.quarantine_count  = _prep_result.quarantine_count
                analytics.quarantine_sample = _prep_result.quarantine_sample
                analytics.cleaning_audit    = _prep_result.cleaning_audit
                async with _async_session() as qa_db:
                    qa_row = await qa_db.get(type(analytics), analytics.id)
                    if qa_row:
                        qa_row.quarantine_count  = _prep_result.quarantine_count
                        qa_row.quarantine_sample = _prep_result.quarantine_sample
                        qa_row.cleaning_audit    = _prep_result.cleaning_audit
                        await qa_db.commit()
                ingest_logger.info(
                    "step", step="5/5", name="cleaning_audit",
                    quarantine_count=_prep_result.quarantine_count,
                    sample_size=len(_prep_result.quarantine_sample),
                    cleaning_audit=_prep_result.cleaning_audit,
                )
            # Trigger parquet conversion whenever the parquet file is missing —
            # regardless of whether the CSV was already preprocessed.  This covers:
            #   • First ingest of a new file
            #   • Re-ingest after parquet files were deleted
            #   • Re-ingest after container was removed and re-connected
            if not analytics.parquet_blob_path:
                # Parquet trigger: flag-gated. Default (celery_parquet_trigger
                # false) keeps today's in-process asyncio.ensure_future. When true,
                # enqueue the durable Celery path instead so the conversion survives
                # a worker restart. The idempotency guard above is unchanged.
                _use_celery_parquet = False
                try:
                    from app.services.ingestion_policy import get_ingestion_policy
                    _use_celery_parquet = bool(
                        get_ingestion_policy().lookup(
                            ("ingestion_features", "celery_parquet_trigger")
                        )
                    )
                except Exception:  # noqa: BLE001 — any failure keeps today's path
                    _use_celery_parquet = False

                if _use_celery_parquet:
                    from app.worker.ingest_tasks import run_scoped_ingest
                    run_scoped_ingest.delay(file_id, "parquet_only")
                else:
                    asyncio.ensure_future(trigger_parquet_conversion(
                        file_id=file_id,
                        blob_path=clean_blob_path,
                        connection_string=container.connection_string,
                        container_name=container.container_name,
                    ))
            else:
                ingest_logger.info("step", step="5/5", name="parquet",
                                   status="skipped", reason="parquet_already_exists",
                                   parquet_path=analytics.parquet_blob_path)

            # ── Schema-dictionary registration for direct .parquet uploads ──
            # CSV/TXT/Excel paths register through parquet_service._run_conversion.
            # A file uploaded as `.parquet` skips that step entirely, so without
            # this branch its definitions would NEVER reach the agent.  Detect
            # and register here using the sample's column profile.
            if ext == parquet_extension(dotted=True):
                try:
                    from app.services.parquet_service import detect_schema_dictionary
                    from app.models.schema_dictionary import SchemaDictionary
                    sd_meta = detect_schema_dictionary(file.name, sample["columns_info"])
                    if sd_meta:
                        async with _async_session() as sd_db:
                            existing = (
                                await sd_db.execute(
                                    select(SchemaDictionary).where(
                                        SchemaDictionary.file_id == file_id
                                    )
                                )
                            ).scalar_one_or_none()
                            if existing:
                                existing.parquet_blob_path = file.blob_path
                                existing.source_blob_path = file.blob_path
                                existing.field_name_col = sd_meta["field_name_col"]
                                existing.description_col = sd_meta["description_col"]
                                existing.notes_col = sd_meta.get("notes_col")
                            else:
                                sd_db.add(SchemaDictionary(
                                    id=str(uuid.uuid4()),
                                    container_id=file.container_id,
                                    file_id=file_id,
                                    parquet_blob_path=file.blob_path,
                                    source_blob_path=file.blob_path,
                                    field_name_col=sd_meta["field_name_col"],
                                    description_col=sd_meta["description_col"],
                                    notes_col=sd_meta.get("notes_col"),
                                ))
                            await sd_db.commit()
                        ingest_logger.info(
                            "schema_dict_registered",
                            file_id=file_id,
                            parquet_path=file.blob_path,
                            field_name_col=sd_meta["field_name_col"],
                            description_col=sd_meta["description_col"],
                            source="parquet_upload",
                        )
                except Exception as sd_exc:
                    ingest_logger.warning(
                        "schema_dict_registration_failed",
                        file_id=file_id,
                        error=str(sd_exc)[:300],
                        source="parquet_upload",
                    )
        except Exception as analytics_exc:
            ingest_logger.warning("step", step="5/5", name="compute_analytics", status="failed",
                                  error=str(analytics_exc)[:300],
                                  duration_ms=_ms(step_start))

        # ── Step 6/6 · Relationship detection ────────────────────────────────
        # Runs AFTER role resolution (step 3b) and analytics (step 5) so
        # column_semantic_roles is committed before the GIN query fires.
        step_start = time.perf_counter()
        ingest_logger.info("step", step="6/6", name="detect_relationships", status="started")
        try:
            from app.services.relationship_detector import detect_relationships  # noqa: PLC0415
            async with _async_session() as rel_db:
                n_rels = await detect_relationships(
                    file_id=file_id,
                    blob_path=clean_blob_path,
                    columns_info=sample["columns_info"],
                    db=rel_db,
                )
            ingest_logger.info("step", step="6/6", name="detect_relationships",
                               status="done",
                               relationships_created=n_rels,
                               duration_ms=_ms(step_start))
        except Exception as rel_exc:
            ingest_logger.warning("step", step="6/6", name="detect_relationships",
                                  status="failed",
                                  error=str(rel_exc)[:300],
                                  duration_ms=_ms(step_start))

        ingest_logger.info("chain_end", outcome="success",
                           filename=file.name,
                           total_duration_ms=_ms(pipeline_start))

        # Invalidate the in-memory catalog so freshly ingested file (and any
        # domain_tag inherited from its folder) is visible to chat without
        # waiting for the 5-minute TTL.
        try:
            from app.agent.catalog_cache import invalidate_catalog_cache
            invalidate_catalog_cache()
        except Exception as _inv_exc:
            ingest_logger.warning("catalog_invalidate_failed", error=str(_inv_exc)[:200])

    except Exception as exc:
        if preprocess_task is not None and not preprocess_task.done():
            preprocess_task.cancel()
        ingest_logger.exception("chain_end", outcome="error",
                                error=str(exc)[:500],
                                total_duration_ms=_ms(pipeline_start))
        try:
            await db.rollback()
            file = await db.get(File, file_id)
            if file:
                file.ingest_status = IngestStatus.FAILED.value
                # Store error in metadata too so the UI can show it
                result = await db.execute(
                    select(FileMetadata).where(FileMetadata.file_id == file_id)
                )
                meta = result.scalar_one_or_none()
                if meta:
                    meta.ingest_error = str(exc)[:1000]
                await db.commit()
        except Exception as inner:
            ingest_logger.error("status_update_failed", error=str(inner)[:300])
