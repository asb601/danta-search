"""
Parquet conversion service — converts CSV to Parquet via streaming PyArrow.

Architecture (zero-disk, bounded RAM):
  Azure CSV blob
    └─ _AzureInputStream  (io.RawIOBase wrapping Azure chunk iterator)
         └─ io.BufferedReader (8 MB read-ahead)
              └─ pa.csv.open_csv()  → BatchedCSVReader  (64 MB batches)
                   └─ pq.ParquetWriter(sink)  (writes each batch as a row group)
                        └─ _ParquetBlobWriter  (buffers → Azure staged blocks)
                             └─ commit_block_list()  (atomic Azure publish)

Peak memory: ~2 × batch_size ≈ 128–256 MB per job, regardless of file size.
No temp files. No DuckDB. Works on any file size without OOM risk.
"""
from __future__ import annotations

import asyncio
import base64
import io
import re
import time

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq
from azure.storage.blob import BlobServiceClient

from app.core.config import get_settings
from app.core.cost_tracker import track_azure_blob
from app.core.logger import ingest_logger
from app.core import metrics
from app.services.ingestion_config import (
    null_tokens,
    parquet_blob_path_for,
    schema_description_tokens,
    schema_field_name_tokens,
    schema_filename_tokens,
    schema_notes_tokens,
)

# ── Azure streaming helpers ──────────────────────────────────────────────────

class _AzureInputStream(io.RawIOBase):
    """io.RawIOBase wrapper around an Azure StorageStreamDownloader chunk iterator.

    Allows pyarrow.csv.open_csv() (which expects a seekable-or-readable file)
    to consume an Azure blob as a streaming HTTP download — no local disk.
    """

    def __init__(self, downloader) -> None:
        self._chunks = downloader.chunks()
        self._buf = b""

    def readable(self) -> bool:
        return True

    def readinto(self, buf: bytearray) -> int:
        while not self._buf:
            try:
                self._buf = next(self._chunks)
            except StopIteration:
                return 0
        n = min(len(buf), len(self._buf))
        buf[:n] = self._buf[:n]
        self._buf = self._buf[n:]
        return n


class _ParquetBlobWriter(io.RawIOBase):
    """Write-only io.RawIOBase sink that stages Azure Block Blob chunks.

    pq.ParquetWriter wraps any io.IOBase subclass in pyarrow.PythonFile, so
    inheriting from io.RawIOBase gives us .closed, .writable() and the standard
    close() protocol for free.

    We buffer to _MIN_BLOCK bytes (8 MB) before staging each block, keeping the
    Azure block count well under the 50 000-block limit even for 100 GB files.
    commit_block_list() is called atomically from close() after ParquetWriter
    has written the footer.

    No disk.  ~8 MB RAM overhead.
    """

    _MIN_BLOCK = max(1, int(get_settings().INGEST_PARQUET_BLOCK_BYTES))

    def __init__(self, blob_client) -> None:
        super().__init__()
        self._bc = blob_client
        self._buf: bytearray = bytearray()
        self._blocks: list[str] = []
        self._pos: int = 0

    # ── io.RawIOBase contract ──────────────────────────────────────────────────

    def writable(self) -> bool:
        return True

    def write(self, data: bytes) -> int:
        self._buf.extend(data)
        self._pos += len(data)
        if len(self._buf) >= self._MIN_BLOCK:
            self._stage_block()
        return len(data)

    def tell(self) -> int:
        return self._pos

    def close(self) -> None:
        if not self.closed:
            self._stage_block()
            if self._blocks:
                self._bc.commit_block_list(self._blocks)
        super().close()

    # ── internal ───────────────────────────────────────────────────────────────

    def _stage_block(self) -> None:
        if not self._buf:
            return
        block_id = base64.b64encode(f"{len(self._blocks):06d}".encode()).decode()
        self._bc.stage_block(block_id, bytes(self._buf))
        self._blocks.append(block_id)
        self._buf.clear()


# ── In-memory progress store (keyed by job_id) ────────────────────────────────
# { job_id: {"phase": "converting"|"uploading", "pct": 0-100} }
_PROGRESS: dict[str, dict] = {}


def get_progress(job_id: str) -> dict | None:
    """Called by the job-status API to get live conversion progress."""
    return _PROGRESS.get(job_id)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)



# ── Schema dictionary detection ───────────────────────────────────────────────
# A file is treated as a data-dictionary ONLY when its filename contains the
# word "schema" (for example schema.csv or fields_schema.csv).
# This is an explicit, opt-in convention — we
# do NOT scan random files looking for dictionary-shaped column names because
# that produces false positives on ordinary lookup tables with name and
# description columns.

# Once a file is identified as a schema by filename, we map its columns to:
#   - field_name_col: the column listing technical field names (e.g. SHKZG)
#   - description_col: the column with the human description
#   - notes_col: optional secondary long-text column
def _norm_col(name: str) -> str:
    """Normalise a column name for pattern matching."""
    return name.lower().replace("-", "_").replace(" ", "_")


def _filename_marks_schema(blob_path: str) -> bool:
    """True if the filename signals this is a data-dictionary upload.

    Match rule: the basename (without directories or extension), tokenised
    on common separators, contains the literal token "schema". This catches:
      - schema.csv
    - schema_fields.csv
    - fields_schema.csv
    - schema-column-map.parquet
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
    return bool(set(tokens) & schema_filename_tokens())


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

    def _pick(tokens: frozenset[str], exclude: set[str]) -> str | None:
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

    field_name_col = _pick(schema_field_name_tokens(), set())
    description_col = _pick(schema_description_tokens(), {field_name_col} if field_name_col else set())

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

    notes_col = _pick(schema_notes_tokens(), {field_name_col, description_col})

    return {
        "field_name_col": field_name_col,
        "description_col": description_col,
        "notes_col": notes_col,
    }


def _profile_from_batch(batch: pa.RecordBatch) -> list[dict]:
    """Compute column profiles from a representative Arrow RecordBatch.

    Returns a list of column profile dicts compatible with the downstream
    schema-dictionary detector and analytics service:
      {"name", "type", "semantic_type", "null_pct",
       "min"?, "max"?, "distinct_count"?, "top_values"?}

    Called on the first streaming batch from the CSV reader, which is a
    representative sample controlled by INGEST_PARQUET_READ_BLOCK_BYTES.
    """
    num_rows = len(batch)
    settings = get_settings()
    top_values_limit = max(0, int(settings.INGEST_PARQUET_TOP_VALUES))
    category_max_ratio = max(0.0, float(settings.INGEST_PARQUET_CATEGORY_MAX_RATIO))
    category_max_distinct = max(1, int(settings.INGEST_PARQUET_CATEGORY_MAX_DISTINCT))
    profiles: list[dict] = []

    for col_name in batch.schema.names:
        col = batch.column(col_name)
        arrow_type = col.type
        null_count = col.null_count

        # ── Semantic type ──────────────────────────────────────────────────────
        if pa.types.is_temporal(arrow_type):
            semantic_type = "date"
        elif (
            pa.types.is_integer(arrow_type)
            or pa.types.is_floating(arrow_type)
            or pa.types.is_decimal(arrow_type)
        ):
            semantic_type = "measure"
        elif pa.types.is_dictionary(arrow_type):
            # auto_dict_encode produced this — low cardinality string column
            semantic_type = "category"
        else:
            semantic_type = "text"

        profile: dict = {
            "name": col_name,
            "type": str(arrow_type),
            "semantic_type": semantic_type,
            "null_pct": round(null_count / num_rows * 100, 1) if num_rows else 0,
        }

        # ── Min / max for numerics and dates ──────────────────────────────────
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

        # ── Cardinality / top-values for strings and categories ───────────────
        if pa.types.is_dictionary(arrow_type):
            try:
                dictionary = col.dictionary
                distinct_count = len(dictionary)
                profile["distinct_count"] = distinct_count
                profile["top_values"] = dictionary[:top_values_limit].to_pylist()
            except Exception:
                pass
        elif pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
            try:
                non_null = col.drop_null()
                if len(non_null) > 0:
                    distinct_count = pc.count_distinct(non_null).as_py()
                    profile["distinct_count"] = distinct_count
                    cardinality_ratio = distinct_count / len(non_null)
                    if cardinality_ratio < category_max_ratio and distinct_count < category_max_distinct:
                        profile["semantic_type"] = "category"
                        vc = non_null.value_counts()
                        sorted_vc = vc.sort_by([("counts", "descending")])
                        profile["top_values"] = sorted_vc["values"][:top_values_limit].to_pylist()
            except Exception:
                pass

        profiles.append(profile)

    return profiles


def _run_conversion(
    blob_path: str,
    connection_string: str,
    container_name: str,
    parquet_blob_path: str,
    job_id: str | None = None,
) -> dict:
    """
    Synchronous CSV→Parquet conversion via streaming PyArrow.
    Runs inside asyncio.to_thread() — never blocks the event loop.

    Data path (zero disk, bounded RAM):
      Azure CSV blob → _AzureInputStream → io.BufferedReader
        → pa.csv.open_csv (64 MB batches)
          → pq.ParquetWriter
            → _ParquetBlobWriter (8 MB staged blocks)
              → Azure Block Blob (atomic commit_block_list)

    Memory: ~2 × batch_size ≈ 128–256 MB regardless of CSV size.
    No tempfiles.  No DuckDB.

    Returns {"parquet_blob_path", "size_bytes", "total_rows",
             "column_profiles", "schema_dict_meta"}
    """
    client = BlobServiceClient.from_connection_string(connection_string)
    src_bc = client.get_blob_client(container=container_name, blob=blob_path)
    dst_bc = client.get_blob_client(container=container_name, blob=parquet_blob_path)

    csv_size_bytes: int = src_bc.get_blob_properties().size

    t_start = time.perf_counter()
    ingest_logger.info(
        "parquet_service", step="convert", status="started",
        blob_path=blob_path, csv_size_mb=round(csv_size_bytes / 1024 / 1024, 1),
    )
    if job_id:
        _PROGRESS[job_id] = {"phase": "converting", "pct": 0}

    settings = get_settings()
    azure_buffer_bytes = max(1, int(settings.INGEST_PARQUET_AZURE_BUFFER_BYTES))
    read_block_bytes = max(1, int(settings.INGEST_PARQUET_READ_BLOCK_BYTES))
    auto_dict_max_cardinality = max(1, int(settings.INGEST_PARQUET_AUTO_DICT_MAX_CARDINALITY))
    progress_max_pct = max(1, min(100, int(settings.INGEST_PARQUET_PROGRESS_MAX_PCT)))
    progress_batch_pct = max(1, int(settings.INGEST_PARQUET_PROGRESS_BATCH_PCT))

    # ── Stream CSV from Azure ─────────────────────────────────────────────────
    downloader = src_bc.download_blob()
    azure_stream = io.BufferedReader(_AzureInputStream(downloader), buffer_size=azure_buffer_bytes)

    reader = pa_csv.open_csv(
        azure_stream,
        read_options=pa_csv.ReadOptions(block_size=read_block_bytes),
        parse_options=pa_csv.ParseOptions(),
        convert_options=pa_csv.ConvertOptions(
            null_values=list(null_tokens()),
            strings_can_be_null=True,
            # Auto-encode low-cardinality string columns as Parquet dictionaries
            # Useful for low-cardinality categorical codes.
            auto_dict_encode=True,
            auto_dict_max_cardinality=auto_dict_max_cardinality,
        ),
    )

    # ── Stream batches → Parquet → Azure staged blocks ────────────────────────
    first_batch: pa.RecordBatch | None = None
    total_rows = 0
    parquet_writer: pq.ParquetWriter | None = None
    blob_writer = _ParquetBlobWriter(dst_bc)

    try:
        for batch_idx, batch in enumerate(reader):
            if first_batch is None:
                first_batch = batch
                parquet_writer = pq.ParquetWriter(
                    blob_writer,
                    batch.schema,
                    compression=settings.INGEST_PARQUET_COMPRESSION,
                    compression_level=max(1, int(settings.INGEST_PARQUET_COMPRESSION_LEVEL)),
                    write_statistics=True,    # row-group min/max → DataFusion predicate pushdown
                    write_page_index=True,    # page-level stats → finer pruning than row-group
                    use_dictionary=True,      # dictionary encoding for low-cardinality string cols
                )
            parquet_writer.write_batch(batch)
            total_rows += len(batch)
            if job_id:
                # Rough progress estimate — each 64 MB batch ≈ 5% of a 1.2 GB CSV
                _PROGRESS[job_id] = {
                    "phase": "converting",
                    "pct": min(progress_max_pct, batch_idx * progress_batch_pct),
                }

        if parquet_writer is not None:
            parquet_writer.close()   # writes Parquet footer into blob_writer buffer
        blob_writer.close()          # flushes remaining buffer + commit_block_list

    except Exception:
        blob_writer._buf.clear()     # discard partial buffer — staged blocks expire
        blob_writer._blocks.clear()  # prevent any accidental commit_block_list
        if parquet_writer is not None:
            try:
                parquet_writer.close()
            except Exception:
                pass
        raise

    convert_ms = _ms(t_start)

    # ── Column profiling (from first batch — representative sample) ───────────
    column_profiles = _profile_from_batch(first_batch) if first_batch is not None else []

    # ── Schema dictionary detection ───────────────────────────────────────────
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

    # Parquet size from blob properties — no local file to stat
    parquet_size_bytes: int = dst_bc.get_blob_properties().size

    ingest_logger.info(
        "parquet_service", step="convert", status="done",
        parquet_size_mb=round(parquet_size_bytes / 1024 / 1024, 1),
        total_rows=total_rows,
        compression_ratio=(
            round(csv_size_bytes / parquet_size_bytes, 2) if parquet_size_bytes else None
        ),
        duration_ms=convert_ms,
    )
    track_azure_blob("download", blob_path, csv_size_bytes, convert_ms)
    track_azure_blob("upload", parquet_blob_path, parquet_size_bytes, convert_ms)

    if job_id:
        _PROGRESS.pop(job_id, None)

    metrics.inc("parquet_conversions")
    metrics.inc("azure_bytes_read", csv_size_bytes)
    metrics.inc("azure_bytes_written", parquet_size_bytes)

    return {
        "parquet_blob_path": parquet_blob_path,
        "size_bytes": parquet_size_bytes,
        "total_rows": total_rows,
        "column_profiles": column_profiles,
        "schema_dict_meta": schema_dict_meta,
    }


async def convert_csv_to_parquet(
    blob_path: str,
    connection_string: str,
    container_name: str,
    job_id: str | None = None,
) -> dict:
    """
    Convert a CSV blob to Parquet using streaming PyArrow (zero disk, bounded RAM).
    Runs synchronous conversion in a thread — never blocks the FastAPI event loop.

    Returns {"parquet_blob_path": str, "size_bytes": int}
    Raises on any failure — caller is responsible for catching and recording the error.
    """
    parquet_blob_path = parquet_blob_path_for(blob_path)

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
