"""
Parquet conversion service — converts CSV to Parquet using DuckDB's az:// reader.

Why DuckDB instead of PyArrow:
  PyArrow required downloading the full CSV to disk first (50+ min on slow connections).
  DuckDB reads directly from az://container/blob.csv via HTTP range requests — no local
  CSV temp file. It also uses all CPU cores for CSV parsing (multi-threaded by default).
  The old "DuckDB az:// bug" (1MB buffer / thousands of HTTP calls) was fixed in DuckDB 1.1.0.
  We're on 1.5.1.

Steps:
  1. DuckDB COPY reads az://container/blob.csv → local Parquet temp file
     (streaming HTTP, parallel reads, ZSTD compression, 100k rows per row group)
  2. Azure SDK uploads Parquet back to blob storage (3-5x smaller than CSV)
  3. Temp Parquet file deleted

Peak memory: DuckDB internal buffers (~256–512MB), regardless of file size.
For a 3GB CSV: total ~2-5 minute conversion + ~1 min upload.
"""
from __future__ import annotations

import asyncio
import os
import re
import tempfile
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

from app.core.cost_tracker import track_azure_blob
from app.core.duckdb_client import _clear_connection, _get_connection
from app.core.logger import ingest_logger
from app.core import metrics

# ── In-memory progress store (keyed by job_id) ────────────────────────────────
# { job_id: {"phase": "converting"|"uploading", "pct": 0-100} }
_PROGRESS: dict[str, dict] = {}


def get_progress(job_id: str) -> dict | None:
    """Called by the job-status API to get live conversion progress."""
    return _PROGRESS.get(job_id)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


# ── Parquet post-processing ────────────────────────────────────────────────────
# Files ≤ MAX_REWRITE_BYTES are fully re-written with PyArrow after DuckDB COPY
# to gain write_statistics, page index, and bloom filters.
# Files > MAX_REWRITE_BYTES skip the full rewrite but still compute profiles
# from the first row group (fast, representative sample).
_MAX_REWRITE_BYTES = 1 * 1024 * 1024 * 1024  # 1 GB


# ── Schema dictionary detection ───────────────────────────────────────────────
# A file is treated as a data-dictionary ONLY when its filename contains the
# word "schema" (e.g. schema.csv, schema_fbl3n.csv, fbl3n_schema.csv,
# schema-bseg-fields.parquet). This is an explicit, opt-in convention — we
# do NOT scan random files looking for dictionary-shaped column names because
# that produces false positives on ordinary lookup/master tables (e.g. a
# vendor master that happens to have name + description columns).

# Once a file is identified as a schema by filename, we map its columns to:
#   - field_name_col: the column listing technical field names (e.g. SHKZG)
#   - description_col: the column with the human description
#   - notes_col: optional secondary long-text column
_FIELD_NAME_TOKENS = frozenset({
    "field", "fieldname", "field_name", "column", "column_name", "col_name",
    "attribute", "technical_name", "tabname", "dataelem", "rollname",
    "element", "fieldid", "field_id", "colname",
})
_DESCRIPTION_TOKENS = frozenset({
    "description", "desc", "short_text", "long_text", "scrtext_m",
    "scrtext_l", "ddtext", "reptext", "label", "meaning",
    "definition", "documentation", "help_text", "medium_text",
    "descr", "description_text", "explanation",
})
_NOTES_TOKENS = frozenset({
    "long_text", "scrtext_l", "documentation", "notes", "remark",
    "extended_description", "detail", "comment",
})


def _norm_col(name: str) -> str:
    """Normalise a column name for pattern matching."""
    return name.lower().replace("-", "_").replace(" ", "_")


def _filename_marks_schema(blob_path: str) -> bool:
    """True if the filename signals this is a data-dictionary upload.

    Match rule: the basename (without directories or extension), tokenised
    on common separators, contains the literal token "schema". This catches:
      - schema.csv
      - schema_fbl3n.csv
      - fbl3n_schema.csv
      - schema-bseg-fields.parquet
      - 2024.10.01_schema_v2.csv
    But NOT a column called "schema_id" inside a regular data file, since
    we never look at columns to decide.
    """
    if not blob_path:
        return False
    # Take the filename only — strip path and extension.
    base = blob_path.rsplit("/", 1)[-1].rsplit(".", 1)[0].lower()
    # Tokenise on _, -, ., space.
    tokens = re.split(r"[_\-.\s]+", base)
    return "schema" in tokens


def detect_schema_dictionary(blob_path: str, profiles: list[dict]) -> dict | None:
    """Return {field_name_col, description_col, notes_col} if the file is a
    data dictionary, else None.

    A file qualifies ONLY if the filename contains "schema" as a token.
    For qualifying files we then map columns by name heuristics.

    If the filename signals "schema" but column names cannot be mapped, we
    log a warning and return None so the file is treated as a normal data
    file — never registered as a dictionary with the wrong columns.
    """
    if not _filename_marks_schema(blob_path):
        return None

    if not profiles:
        return None

    col_names = [p["name"] for p in profiles]
    normed = {n: _norm_col(n) for n in col_names}

    def _pick(tokens: frozenset, exclude: set[str]) -> str | None:
        # Exact match first.
        for original, norm in normed.items():
            if original in exclude:
                continue
            if norm in tokens:
                return original
        # Substring match fallback.
        for original, norm in normed.items():
            if original in exclude:
                continue
            if any(t in norm for t in tokens):
                return original
        return None

    field_name_col = _pick(_FIELD_NAME_TOKENS, set())
    description_col = _pick(_DESCRIPTION_TOKENS, {field_name_col} if field_name_col else set())

    if not field_name_col or not description_col:
        ingest_logger.warning(
            "schema_dict_columns_not_recognised",
            blob_path=blob_path,
            columns=col_names[:30],
            hint=(
                "Filename marks this as a schema dictionary but the column "
                "names do not match the expected (field_name, description) "
                "convention. Add a column whose name contains 'field' and "
                "another whose name contains 'description' to register it."
            ),
        )
        return None

    notes_col = _pick(_NOTES_TOKENS, {field_name_col, description_col})

    return {
        "field_name_col": field_name_col,
        "description_col": description_col,
        "notes_col": notes_col,
    }


def _profile_and_rewrite(tmp_parquet_path: str) -> list[dict]:
    """Compute column profiles and rewrite the local Parquet with optimal settings.

    Returns a list of column profile dicts, one per column:
      {"name", "type", "semantic_type", "null_pct",
       "min"?, "max"?, "distinct_count"?, "top_values"?}

    Side effect: overwrites tmp_parquet_path in place (only if file ≤ _MAX_REWRITE_BYTES).
    """
    file_size = os.path.getsize(tmp_parquet_path)
    pf = pq.ParquetFile(tmp_parquet_path)

    # Profile from first row group only (fast, ~representative for most files)
    sample = pf.read_row_group(0)
    num_sample_rows = len(sample)

    profiles: list[dict] = []
    bloom_cols: list[str] = []

    for col_name in sample.column_names:
        col = sample.column(col_name)
        arrow_type_str = str(col.type)
        null_count = col.null_count

        # ── Semantic type detection ────────────────────────────────────────────
        if pa.types.is_temporal(col.type):
            semantic_type = "date"
        elif (pa.types.is_integer(col.type)
              or pa.types.is_floating(col.type)
              or pa.types.is_decimal(col.type)):
            semantic_type = "measure"
        else:
            semantic_type = "text"

        profile: dict = {
            "name": col_name,
            "type": arrow_type_str,
            "semantic_type": semantic_type,
            "null_pct": round(null_count / num_sample_rows * 100, 1) if num_sample_rows else 0,
        }

        # ── Min / max for numeric and date columns ─────────────────────────────
        if semantic_type in ("measure", "date"):
            try:
                min_v = pc.min(col).as_py()
                max_v = pc.max(col).as_py()
                if min_v is not None:
                    profile["min"] = str(min_v) if semantic_type == "date" else min_v
                if max_v is not None:
                    profile["max"] = str(max_v) if semantic_type == "date" else max_v
            except Exception:
                pass

        # ── Cardinality for string columns ─────────────────────────────────────
        if pa.types.is_large_string(col.type) or pa.types.is_string(col.type):
            try:
                non_null = col.drop_null()
                if len(non_null) > 0:
                    distinct_count = pc.count_distinct(non_null).as_py()
                    profile["distinct_count"] = distinct_count
                    cardinality_ratio = distinct_count / len(non_null)
                    if cardinality_ratio < 0.05 and distinct_count < 5000:
                        profile["semantic_type"] = "category"
                        bloom_cols.append(col_name)
                        # Top 10 most frequent values
                        vc = non_null.value_counts()
                        sorted_vc = vc.sort_by([("counts", "descending")])
                        profile["top_values"] = sorted_vc["values"][:10].to_pylist()
            except Exception:
                pass

        profiles.append(profile)

    # ── Rewrite with optimal Parquet settings (only for manageable file sizes) ─
    if file_size <= _MAX_REWRITE_BYTES:
        full_table = pq.read_table(tmp_parquet_path)
        write_kwargs: dict = dict(
            compression="zstd",
            compression_level=3,
            row_group_size=1_000_000,
            write_statistics=True,
            write_page_index=True,
            use_dictionary=True,
        )
        if bloom_cols:
            write_kwargs["bloom_filter_columns"] = bloom_cols
        # Write to a sibling temp file then atomically replace the original.
        # Avoids a corrupt file if the process dies mid-write.
        tmp_rewrite = tmp_parquet_path + ".rewrite"
        try:
            pq.write_table(full_table, tmp_rewrite, **write_kwargs)
            os.replace(tmp_rewrite, tmp_parquet_path)
        finally:
            if os.path.exists(tmp_rewrite):
                try:
                    os.unlink(tmp_rewrite)
                except OSError:
                    pass

    return profiles


def _run_conversion(
    blob_path: str,
    connection_string: str,
    container_name: str,
    parquet_blob_path: str,
    job_id: str | None = None,
) -> dict:
    """
    Synchronous conversion. Runs inside asyncio.to_thread() — never blocks the event loop.

    Steps:
      1. DuckDB reads CSV from az://container/blob.csv → local Parquet temp file
      2. Azure SDK uploads Parquet back to Azure Blob Storage
      3. Temp Parquet file deleted

    Returns {"parquet_blob_path": str, "size_bytes": int}
    """
    tmp_parquet_path = None

    try:
        azure_csv_path = f"az://{container_name}/{blob_path}"
        client = BlobServiceClient.from_connection_string(connection_string)

        # Get CSV size upfront for cost tracking
        csv_size_bytes: int = (
            client.get_blob_client(container=container_name, blob=blob_path)
            .get_blob_properties()
            .size
        )

        # ── Step 1: DuckDB converts CSV from Azure → local Parquet temp file ──
        t = time.perf_counter()
        ingest_logger.info("parquet_service", step="convert", status="started",
                           blob_path=blob_path, csv_size_mb=round(csv_size_bytes / 1024 / 1024, 1))

        if job_id:
            _PROGRESS[job_id] = {"phase": "converting", "pct": 0}

        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_parquet_path = tmp.name

        conn = _get_connection(connection_string)
        safe_csv = azure_csv_path.replace("'", "''")
        safe_parquet = tmp_parquet_path.replace("'", "''")

        try:
            conn.execute(f"""
                COPY (
                    SELECT * FROM read_csv_auto(
                        '{safe_csv}',
                        null_padding=true,
                        ignore_errors=true,
                        nullstr=['', 'NULL', 'null', 'N/A', 'n/a', 'NA', 'na',
                                 'None', 'none', 'NaN', 'nan', '-', 'TBD', 'tbd']
                    )
                )
                TO '{safe_parquet}'
                (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000)
            """)
        except Exception:
            _clear_connection(connection_string)
            raise

        parquet_size_bytes = os.path.getsize(tmp_parquet_path)

        # Row count from Parquet metadata — cheap, no full scan needed
        pf = pq.ParquetFile(tmp_parquet_path)
        total_rows = pf.metadata.num_rows
        num_row_groups = pf.metadata.num_row_groups

        convert_ms = _ms(t)
        ingest_logger.info("parquet_service", step="convert", status="done",
                           parquet_size_mb=round(parquet_size_bytes / 1024 / 1024, 1),
                           total_rows=total_rows,
                           row_groups=num_row_groups,
                           compression_ratio=round(csv_size_bytes / parquet_size_bytes, 2),
                           duration_ms=convert_ms)
        track_azure_blob("download", blob_path, csv_size_bytes, convert_ms)

        # ── Step 1b: Profile columns + rewrite with statistics / bloom filters ─
        t_profile = time.perf_counter()
        column_profiles = _profile_and_rewrite(tmp_parquet_path)
        # Rewrite may change size — refresh
        parquet_size_bytes = os.path.getsize(tmp_parquet_path)

        # ── Step 1c: Detect schema dictionary ────────────────────────────────
        # Filename-based: the upload must contain "schema" as a token in the
        # filename (e.g. schema_fbl3n.csv). We never auto-classify a regular
        # data file as a dictionary based on column names alone.
        schema_dict_meta = detect_schema_dictionary(blob_path, column_profiles)
        if schema_dict_meta:
            ingest_logger.info(
                "parquet_service",
                step="schema_dict_detected",
                field_name_col=schema_dict_meta["field_name_col"],
                description_col=schema_dict_meta["description_col"],
                notes_col=schema_dict_meta.get("notes_col"),
                parquet_blob_path=parquet_blob_path,
            )

        ingest_logger.info("parquet_service", step="rewrite", status="done",
                           bloom_cols=len([p for p in column_profiles if "top_values" in p]),
                           profiles=len(column_profiles),
                           duration_ms=_ms(t_profile))

        # ── Step 2: upload Parquet back to Azure ──────────────────────────────
        t = time.perf_counter()
        ingest_logger.info("parquet_service", step="upload", status="started",
                           parquet_blob_path=parquet_blob_path)
        if job_id:
            _PROGRESS[job_id] = {"phase": "uploading", "pct": 0}

        parquet_blob_client = client.get_blob_client(container=container_name, blob=parquet_blob_path)
        with open(tmp_parquet_path, "rb") as f:
            parquet_blob_client.upload_blob(f, overwrite=True)

        ingest_logger.info("parquet_service", step="upload", status="done",
                           size_mb=round(parquet_size_bytes / 1024 / 1024, 1),
                           duration_ms=_ms(t))
        track_azure_blob("upload", parquet_blob_path, parquet_size_bytes, _ms(t))

        if job_id:
            _PROGRESS.pop(job_id, None)  # clean up — job is done

        metrics.inc("parquet_conversions")
        metrics.inc("azure_bytes_read", csv_size_bytes)
        metrics.inc("azure_bytes_written", parquet_size_bytes)
        return {
            "parquet_blob_path": parquet_blob_path,
            "size_bytes": parquet_size_bytes,
            "total_rows": total_rows,
            "column_profiles": column_profiles,
            # Present only if the file was identified as a data dictionary.
            "schema_dict_meta": schema_dict_meta,
        }

    finally:
        if tmp_parquet_path and os.path.exists(tmp_parquet_path):
            try:
                os.unlink(tmp_parquet_path)
            except OSError:
                pass


async def convert_csv_to_parquet(
    blob_path: str,
    connection_string: str,
    container_name: str,
    job_id: str | None = None,
) -> dict:
    """
    Convert a CSV blob to Parquet using DuckDB's az:// reader.
    Runs synchronous conversion in a thread — never blocks the FastAPI event loop.

    Returns {"parquet_blob_path": str, "size_bytes": int}
    Raises on any failure — caller is responsible for catching and recording the error.
    """
    parquet_blob_path = blob_path.rsplit(".", 1)[0] + ".parquet"

    start = time.perf_counter()
    ingest_logger.info("parquet_service", operation="convert_csv_to_parquet",
                       status="started", blob_path=blob_path,
                       target_path=parquet_blob_path)

    result = await asyncio.to_thread(
        _run_conversion,
        blob_path,
        connection_string,
        container_name,
        parquet_blob_path,
        job_id,
    )

    ingest_logger.info("parquet_service", operation="convert_csv_to_parquet",
                       status="done", total_duration_ms=_ms(start),
                       size_bytes=result.get("size_bytes"))

    return result
