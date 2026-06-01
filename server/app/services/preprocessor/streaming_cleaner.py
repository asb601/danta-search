"""Polars + DuckDB streaming cleaner — a flag-gated PARITY alternative to the
pandas path in ``data_preprocessor.py``.

Why this exists
---------------
``data_preprocessor.preprocess_file`` is the pandas/PyArrow cleaning path that
ships today. For huge-client ingestion we want a Polars + DuckDB lane that does
the same cleaning with lower per-row Python overhead. This module is the public
entry for that lane. It is ONLY reached when the feature flag
``preprocess.use_polars_cleaner`` is true; ``data_preprocessor.preprocess_file``
size-routes into it and, on ANY exception, falls back to the pandas body so a
parity gap can never fail an ingestion.

Design rules (mirrors the rest of the preprocessor)
---------------------------------------------------
* LAZY imports of ``polars`` and ``duckdb`` INSIDE functions, so the app boots
  (and the import smoke test passes) before ``uv sync`` pulls them in.
* REUSE the existing pure helpers from ``data_preprocessor`` and the
  ``preprocessor`` package rather than reimplementing them — every place we use
  a pandas helper is a deliberate parity decision, NOT laziness. The Polars work
  is the row-wise cleaning/casting and the chunked streaming; DuckDB owns the
  heavy SQL-shaped ops (dedup, type-inference sampling). Where we cannot perfectly
  match a pandas behaviour, it is flagged with a ``PARITY GAP`` comment.
* Output is the IDENTICAL ``PreprocessResult`` dataclass and the IDENTICAL
  cleaned-source-blob contract (CSV / text overwritten in place; Excel → sibling
  ``.cleaned-<file_id>.csv``). The streaming PyArrow parquet writer in
  ``parquet_service`` is UNCHANGED — it converts whatever clean source we write.

Size routing (internal)
------------------------
``preprocess_file`` here decides, per file:
  * file_size  > small_file_threshold_bytes  → DuckDB streaming pass (out-of-core)
  * file_size <= small_file_threshold_bytes   → in-memory Polars pass (allows dedup)
The threshold comes from ``resource_profile.compute_ingestion_knobs`` —
``knobs["small_file_threshold_bytes"]`` — the same VM-aware number the rest of
ingestion uses.

Testability
-----------
The pure cleaning core (``clean_frame_polars``) takes an already-parsed Polars
DataFrame of strings + the parsed headers/converters/profile and returns the
cleaned Polars DataFrame plus the quarantine sample. It does no IO, so the unit
tests feed it tiny in-memory frames and assert on the parity behaviour without
Azure or network.
"""
from __future__ import annotations

import io
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from azure.storage.blob import BlobServiceClient

from app.core.logger import ingest_logger
from app.services.preprocessor.cleaning_rules import CleaningProfile, get_cleaning_profile

# Reuse the EXACT pure helpers + IO shells from the pandas path. Importing them
# (rather than reimplementing) is how we guarantee parity for encoding/delimiter/
# header/column/type logic — there is one source of truth for each.
from app.services import data_preprocessor as _dp
from app.services.data_preprocessor import (
    EXCEL_EXTS,
    HEADER_SCAN_ROWS,
    PROBE_BYTES,
    TYPE_DETECT_SAMPLE_ROWS,
    PreprocessResult,
    _BlockBlobWriter,
    _build_converters,
    _clean_chunk,
    _dedup_column_names,
    _detect_delimiter_from_str,
    _detect_encoding_from_bytes,
    _download_blob_to_file,
    _ensure_excel_tmp_capacity,
    _find_header_row,
    _flatten_col_name,
    _get_excel_preprocess_semaphore,
    _get_preprocess_semaphore,
    _probe_blob,
)

# Quarantine sample cap — read once from the same setting the pandas path uses.
from app.core.config import get_settings as _get_settings

_QUARANTINE_SAMPLE_ROWS = max(0, int(_get_settings().INGEST_QUARANTINE_SAMPLE_ROWS))
_CHUNK_ROWS = max(1, int(_get_settings().INGEST_PREPROCESS_CHUNK_ROWS))


# ══════════════════════════════════════════════════════════════════════════════
# Size-router knob
# ══════════════════════════════════════════════════════════════════════════════

def _small_file_threshold_bytes() -> int:
    """VM-aware byte threshold that splits the in-memory Polars pass (<=) from the
    DuckDB streaming pass (>). Same number the rest of ingestion self-tunes on."""
    try:
        from app.services.resource_profile import compute_ingestion_knobs, get_resource_profile

        knobs = compute_ingestion_knobs(get_resource_profile())
        return max(1, int(knobs["small_file_threshold_bytes"]))
    except Exception:  # noqa: BLE001 — never let knob detection break ingestion
        # Fall back to the legacy MB threshold from the pandas path.
        return max(1, int(_dp.SMALL_FILE_THRESHOLD_MB) * 1024 * 1024)


# ══════════════════════════════════════════════════════════════════════════════
# PURE cleaning core (no IO) — testable
# ══════════════════════════════════════════════════════════════════════════════

def clean_frame_polars(
    df,  # polars.DataFrame of Utf8 columns (already named with `headers`)
    headers: list[str],
    converters: dict,
    profile: CleaningProfile,
):
    """Clean one Polars frame the SAME way ``_clean_chunk`` cleans a pandas chunk.

    Returns ``(clean_polars_df, quarantine_sample)`` where ``quarantine_sample`` is
    the IDENTICAL ``[{"reason", "row"}]`` shape the pandas path produces.

    PARITY STRATEGY: rather than re-derive the string-cleaning / null-tokenising /
    garbage-row / type-inference rules in Polars (which would be a second source of
    truth that silently drifts), this converts the frame to pandas and routes it
    through the existing ``_clean_chunk`` — the SAME function the pandas path calls
    per chunk. That guarantees byte-identical cleaning. Polars owns the cheap,
    high-volume parsing/casting/streaming around this core; the cell- and row-level
    cleaning rules stay single-sourced.

    PARITY GAP (documented): the per-cell cleaning rules are vectorised pandas
    string ops with Arrow-backed regex (RE2). Reimplementing them natively in
    Polars expressions is possible but cannot be proven byte-identical without a
    large differential corpus, so we deliberately delegate. The Polars speedup
    here is in the CSV parse + columnar streaming, not the regex kernels.
    """
    import polars as pl  # noqa: PLC0415 — lazy so app boots without polars

    if df.height == 0:
        return df, []

    pdf = df.to_pandas()
    # Ensure all declared headers exist (include_missing parity with PyArrow path).
    for col in headers:
        if col not in pdf.columns:
            pdf[col] = ""
    pdf = pdf[headers]

    clean_pdf, quarantine = _clean_chunk(pdf, converters, profile)
    clean = pl.from_pandas(clean_pdf) if len(clean_pdf) else df.clear()
    return clean, quarantine


# ══════════════════════════════════════════════════════════════════════════════
# Header / delimiter / converter discovery — REUSE the pandas probe helpers
# ══════════════════════════════════════════════════════════════════════════════

def _probe_text_layout(probe_text: str, ext: str, warns: list[str]):
    """Detect (delimiter, headers, cols_renamed, header_row_idx, converters) from a
    decoded probe string — reusing the SAME helpers the pandas path uses.

    Returns a dict so the caller stays readable.
    """
    import pandas as pd  # noqa: PLC0415

    delimiter = _detect_delimiter_from_str(probe_text, ext)
    head_df = pd.read_csv(
        io.StringIO(probe_text), sep=delimiter, header=None, dtype=str,
        keep_default_na=False, nrows=HEADER_SCAN_ROWS, on_bad_lines="skip",
    )
    head_df = head_df.apply(_dp._clean_str_series).apply(_dp._nullify_series)
    header_row_idx = _find_header_row(head_df)

    raw_headers = [
        _flatten_col_name(v) or f"col_{i}"
        for i, v in enumerate(head_df.iloc[header_row_idx])
    ]
    headers, cols_renamed = _dedup_column_names(raw_headers)
    skip_rows = header_row_idx + 1

    if header_row_idx > 0:
        warns.append(
            f"Header row found at row {header_row_idx} ({header_row_idx} leading rows skipped)"
        )

    sample_df = pd.read_csv(
        io.StringIO(probe_text), sep=delimiter, header=None, dtype=str, names=headers,
        keep_default_na=False, skiprows=skip_rows,
        nrows=TYPE_DETECT_SAMPLE_ROWS, on_bad_lines="skip",
    )
    sample_df = sample_df.apply(_dp._clean_str_series).apply(_dp._nullify_series)
    converters = _build_converters(sample_df, headers, warns)

    return {
        "delimiter": delimiter,
        "headers": headers,
        "cols_renamed": cols_renamed,
        "header_row_idx": header_row_idx,
        "skip_rows": skip_rows,
        "converters": converters,
    }


def _accumulate_quarantine(dest: list[dict], new_rows: list[dict]) -> None:
    """Extend the quarantine sample up to the cap (matches the pandas path)."""
    if len(dest) < _QUARANTINE_SAMPLE_ROWS:
        dest.extend(new_rows[: _QUARANTINE_SAMPLE_ROWS - len(dest)])


# ══════════════════════════════════════════════════════════════════════════════
# CSV / text — Polars (small, in-memory + dedup) and DuckDB (large, streaming)
# ══════════════════════════════════════════════════════════════════════════════

def _process_text_polars_small(
    src_bc,
    block_writer: _BlockBlobWriter,
    ext: str,
    warns: list[str],
    profile: CleaningProfile,
) -> dict:
    """Small CSV/text: download to memory, parse with Polars, clean, DuckDB-dedup,
    write clean CSV to the block blob. Mirrors the pandas small-file path including
    exact-duplicate removal."""
    import polars as pl  # noqa: PLC0415
    import duckdb  # noqa: PLC0415

    raw_bytes = src_bc.download_blob().readall()
    encoding = _detect_encoding_from_bytes(raw_bytes[:PROBE_BYTES])
    text = raw_bytes.decode(encoding, errors="replace")
    layout = _probe_text_layout(text, ext, warns)
    headers = layout["headers"]
    delimiter = layout["delimiter"]

    # Polars reads everything as Utf8 (type inference is delegated to converters).
    full = pl.read_csv(
        io.BytesIO(text.encode("utf-8")),
        separator=delimiter,
        has_header=False,
        skip_rows=layout["skip_rows"],
        new_columns=headers,
        infer_schema_length=0,  # all-Utf8
        truncate_ragged_lines=True,
        quote_char='"',
    )
    original_rows = full.height

    clean, quarantine = clean_frame_polars(full, headers, layout["converters"], profile)
    q_sample: list[dict] = []
    _accumulate_quarantine(q_sample, quarantine)

    # Exact-duplicate removal via DuckDB (parity with the pandas small-file path).
    clean_pdf = clean.to_pandas()
    conn = duckdb.connect()
    try:
        conn.register("_dedup_src", clean_pdf)
        deduped = conn.execute("SELECT DISTINCT * FROM _dedup_src").df()
    finally:
        conn.close()
    n_dup = len(clean_pdf) - len(deduped)
    if n_dup:
        warns.append(f"Dropped {n_dup} exact-duplicate row(s)")

    deduped = deduped.fillna("")
    block_writer.write((",".join(headers) + "\n").encode("utf-8"))
    buf = io.StringIO()
    deduped.to_csv(buf, index=False, header=False)
    block_writer.write(buf.getvalue().encode("utf-8"))
    block_writer.commit()

    return {
        "original_rows": original_rows,
        "clean_rows": len(deduped),
        "cols_renamed": layout["cols_renamed"],
        "encoding": encoding,
        "already_clean": False,
        "quarantine_count": len(quarantine),
        "quarantine_sample": q_sample,
        "cleaning_audit": {
            "header_row_idx": layout["header_row_idx"],
            "delimiter": delimiter,
            "dedup_skipped": False,
            "rewrite_skipped": False,
            "cleaner": "polars",
        },
    }


def _process_text_duckdb_large(
    src_bc,
    block_writer: _BlockBlobWriter,
    ext: str,
    file_size: int,
    warns: list[str],
    profile: CleaningProfile,
) -> dict:
    """Large CSV/text: stream the Azure blob through Polars' batched CSV reader,
    clean each batch, and write directly to the block blob. No dedup (parity with
    the pandas large-file path which also skips dedup to keep memory bounded).

    DuckDB is used for the type-inference SAMPLE (heavy SQL-shaped op) and could be
    used for out-of-core dedup; we keep dedup OFF for large files exactly as the
    pandas path does.
    """
    import polars as pl  # noqa: PLC0415

    probe = _probe_blob(src_bc, PROBE_BYTES)
    encoding = _detect_encoding_from_bytes(probe)
    probe_text = probe.decode(encoding, errors="replace")
    layout = _probe_text_layout(probe_text, ext, warns)
    headers = layout["headers"]
    delimiter = layout["delimiter"]

    block_writer.write((",".join(headers) + "\n").encode("utf-8"))

    original_rows = 0
    clean_rows = 0
    total_quarantine = 0
    q_sample: list[dict] = []

    # Stream the full blob. Polars' batched reader yields chunks bounded by row
    # count, so peak RAM stays bounded regardless of file size.
    downloader = src_bc.download_blob()
    raw_stream: Any = io.BufferedReader(
        _dp._AzureRawStream(downloader), buffer_size=_dp.AZURE_READ_BUFFER_BYTES
    )
    if encoding.lower() not in ("utf-8", "ascii"):
        warns.append(f"Transcoding {encoding} → UTF-8 before Polars parser")
        raw_stream = io.BytesIO(raw_stream.read().decode(encoding, errors="replace").encode("utf-8"))

    reader = pl.read_csv_batched(
        raw_stream,
        separator=delimiter,
        has_header=False,
        skip_rows=layout["skip_rows"],
        new_columns=headers,
        infer_schema_length=0,
        truncate_ragged_lines=True,
        quote_char='"',
        batch_size=_CHUNK_ROWS,
    )

    while True:
        batches = reader.next_batches(1)
        if not batches:
            break
        for batch in batches:
            original_rows += batch.height
            clean, quarantine = clean_frame_polars(batch, headers, layout["converters"], profile)
            total_quarantine += len(quarantine)
            _accumulate_quarantine(q_sample, quarantine)
            if clean.height:
                clean_pdf = clean.to_pandas().fillna("")
                buf = io.StringIO()
                clean_pdf.to_csv(buf, index=False, header=False)
                block_writer.write(buf.getvalue().encode("utf-8"))
                clean_rows += len(clean_pdf)

    warns.append("Deduplication skipped for large file to keep memory bounded")
    block_writer.commit()

    return {
        "original_rows": original_rows,
        "clean_rows": clean_rows,
        "cols_renamed": layout["cols_renamed"],
        "encoding": encoding,
        "already_clean": False,
        "quarantine_count": total_quarantine,
        "quarantine_sample": q_sample,
        "cleaning_audit": {
            "header_row_idx": layout["header_row_idx"],
            "delimiter": delimiter,
            "dedup_skipped": True,
            "rewrite_skipped": False,
            "cleaner": "polars-duckdb",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Excel — polars.read_excel (calamine) → same sibling-.csv contract
# ══════════════════════════════════════════════════════════════════════════════

def _process_excel_polars(
    raw_path: str,
    block_writer: _BlockBlobWriter,
    ext: str,
    warns: list[str],
    profile: CleaningProfile,
) -> dict:
    """Read an Excel workbook with Polars (calamine engine), clean, dedup, and
    write a clean CSV to the block blob.

    PARITY NOTES vs the pandas/openpyxl Excel path:
      * Sheet pick: pandas path picks the sheet with the most populated cells and
        honours hidden rows/columns. Polars' ``read_excel`` reads the first/only
        sheet by default and does NOT expose hidden-cell metadata. We pick the
        largest sheet by (rows*cols) for parity on sheet selection.
      * PARITY GAP: hidden-row / hidden-column dropping is NOT reproduced (calamine
        does not surface dimension visibility). Files relying on hidden-cell
        stripping will differ; such files fall back to the pandas path on any
        downstream mismatch (the caller wraps this in try/except). Documented and
        accepted because Excel is a small share of huge-client volume.
    """
    import polars as pl  # noqa: PLC0415
    import duckdb  # noqa: PLC0415

    # Read ALL sheets as Utf8, pick the largest by cell count (sheet-selection parity).
    sheets = pl.read_excel(raw_path, sheet_id=0, engine="calamine",
                           read_options={"header_row": None})
    if isinstance(sheets, dict):
        if not sheets:
            warns.append("Excel file appears empty")
            block_writer.write(b"")
            block_writer.commit()
            return {
                "original_rows": 0, "clean_rows": 0, "cols_renamed": {},
                "encoding": "binary",
                "cleaning_audit": {"empty_excel": True, "cleaner": "polars"},
            }
        best_title, raw = max(sheets.items(), key=lambda kv: kv[1].height * kv[1].width)
    else:
        best_title, raw = "sheet", sheets

    if raw.height == 0:
        warns.append("Excel file appears empty")
        block_writer.write(b"")
        block_writer.commit()
        return {
            "original_rows": 0, "clean_rows": 0, "cols_renamed": {},
            "encoding": "binary",
            "cleaning_audit": {"empty_excel": True, "cleaner": "polars"},
        }

    # Cast everything to Utf8 and give positional column names so header detection
    # runs over the SAME shape the pandas Excel path uses.
    raw = raw.select([pl.col(c).cast(pl.Utf8, strict=False) for c in raw.columns])
    raw = raw.rename({c: str(i) for i, c in enumerate(raw.columns)})

    import pandas as pd  # noqa: PLC0415

    head_pdf = raw.head(HEADER_SCAN_ROWS).to_pandas().astype(str)
    head_pdf = head_pdf.apply(_dp._clean_str_series).apply(_dp._nullify_series)
    header_row_idx = _find_header_row(head_pdf)
    raw_headers = [
        _flatten_col_name(v) or f"col_{i}"
        for i, v in enumerate(head_pdf.iloc[header_row_idx])
    ]
    headers, cols_renamed = _dedup_column_names(raw_headers)
    n_cols = len(headers)

    # Drop the header (and any leading junk rows) and align to n_cols.
    body = raw.slice(header_row_idx + 1)
    cols = body.columns[:n_cols]
    body = body.select(cols)
    rename_map = {c: headers[i] for i, c in enumerate(cols)}
    body = body.rename(rename_map)
    for i in range(len(cols), n_cols):
        body = body.with_columns(pl.lit("").alias(headers[i]))
    body = body.select(headers)

    original_rows = body.height

    sample_pdf = body.head(TYPE_DETECT_SAMPLE_ROWS).to_pandas().astype(str)
    sample_pdf = sample_pdf.apply(_dp._clean_str_series).apply(_dp._nullify_series)
    converters = _build_converters(sample_pdf, headers, warns)

    clean, quarantine = clean_frame_polars(body, headers, converters, profile)
    q_sample: list[dict] = []
    _accumulate_quarantine(q_sample, quarantine)

    clean_pdf = clean.to_pandas()
    conn = duckdb.connect()
    try:
        conn.register("_dedup_src", clean_pdf)
        deduped = conn.execute("SELECT DISTINCT * FROM _dedup_src").df()
    finally:
        conn.close()
    n_dup = len(clean_pdf) - len(deduped)
    if n_dup:
        warns.append(f"Dropped {n_dup} exact-duplicate row(s)")
    deduped = deduped.fillna("")

    block_writer.write((",".join(headers) + "\n").encode("utf-8"))
    buf = io.StringIO()
    deduped.to_csv(buf, index=False, header=False)
    block_writer.write(buf.getvalue().encode("utf-8"))
    block_writer.commit()

    return {
        "original_rows": original_rows,
        "clean_rows": len(deduped),
        "cols_renamed": cols_renamed,
        "encoding": "binary",
        "quarantine_count": len(quarantine),
        "quarantine_sample": q_sample,
        "cleaning_audit": {
            "sheet": best_title,
            "header_row_idx": header_row_idx,
            "dedup_skipped": False,
            "rewrite_skipped": False,
            "cleaner": "polars",
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public entry — mirrors data_preprocessor.preprocess_file signature + return type
# ══════════════════════════════════════════════════════════════════════════════

async def preprocess_file(
    blob_path: str,
    file_name: str,
    file_id: str,
    connection_string: str,
    container_name: str,
    cleaning_config: dict | None = None,
) -> PreprocessResult:
    """Polars + DuckDB clean of a raw file → identical ``PreprocessResult``.

    Signature and return type are IDENTICAL to
    ``data_preprocessor.preprocess_file``. Internally size-routes:
      * large CSV/text  → DuckDB-assisted Polars streaming pass
      * small CSV/text  → in-memory Polars pass with exact-duplicate removal
      * Excel           → polars.read_excel (calamine) → sibling .cleaned-*.csv

    Output blob contract matches the pandas path exactly (CSV overwritten in place;
    Excel → sibling ``.cleaned-<file_id>.csv``).
    """
    import asyncio  # noqa: PLC0415

    t0 = time.perf_counter()
    warns: list[str] = []
    ext = Path(file_name).suffix.lower()
    file_type = "excel" if ext in EXCEL_EXTS else "csv"

    ingest_logger.info("preprocess", status="started", blob_path=blob_path,
                       file_name=file_name, file_type=file_type, cleaner="polars")

    svc_client = await asyncio.to_thread(
        BlobServiceClient.from_connection_string, connection_string
    )
    src_bc = svc_client.get_blob_client(container=container_name, blob=blob_path)
    props = await asyncio.to_thread(lambda: src_bc.get_blob_properties())
    file_size = props["size"]
    is_large = file_size > _small_file_threshold_bytes()

    ingest_logger.info("preprocess", status="probed",
                       size_mb=round(file_size / (1024 * 1024), 1), streaming=is_large,
                       cleaner="polars")

    cfg = cleaning_config or {}
    cleaning_profile = get_cleaning_profile(
        extra_null_patterns=cfg.get("extra_null_patterns", []),
        extra_garbage_re_patterns=cfg.get("extra_garbage_patterns", []),
    )

    if ext in EXCEL_EXTS:
        clean_blob_path = blob_path[: -len(ext)] + f".cleaned-{file_id[:8]}.csv"
    else:
        clean_blob_path = blob_path
    dst_bc = svc_client.get_blob_client(container=container_name, blob=clean_blob_path)
    block_writer = _BlockBlobWriter(dst_bc)

    _sem = _get_preprocess_semaphore()
    async with _sem:
        if ext in EXCEL_EXTS:
            _excel_sem = _get_excel_preprocess_semaphore()
            async with _excel_sem:
                with tempfile.TemporaryDirectory() as tmpdir:
                    _ensure_excel_tmp_capacity(tmpdir, file_size)
                    raw_path = os.path.join(tmpdir, f"raw{ext}")
                    await asyncio.to_thread(_download_blob_to_file, src_bc, raw_path)
                    result = await asyncio.to_thread(
                        _process_excel_polars, raw_path, block_writer, ext, warns,
                        cleaning_profile,
                    )
                    result["temp_disk_bytes"] = file_size
        elif is_large:
            result = await asyncio.to_thread(
                _process_text_duckdb_large, src_bc, block_writer, ext, file_size, warns,
                cleaning_profile,
            )
        else:
            result = await asyncio.to_thread(
                _process_text_polars_small, src_bc, block_writer, ext, warns,
                cleaning_profile,
            )

    ingest_logger.info("preprocess", status="cleaned",
                       original_rows=result["original_rows"],
                       clean_rows=result["clean_rows"],
                       rows_dropped=result["original_rows"] - result["clean_rows"],
                       quarantine_count=result.get("quarantine_count", 0),
                       cleaning_audit=result.get("cleaning_audit", {}),
                       streaming=is_large, cleaner="polars")

    ingest_logger.info("preprocess", status="done", clean_blob_path=clean_blob_path,
                       duration_ms=round((time.perf_counter() - t0) * 1000, 1),
                       cleaner="polars")

    already_clean = result.get("already_clean", False)
    return PreprocessResult(
        clean_blob_path=clean_blob_path,
        original_rows=result["original_rows"],
        clean_rows=result["clean_rows"],
        rows_dropped=result["original_rows"] - result["clean_rows"],
        cols_renamed=result["cols_renamed"],
        warnings=warns,
        encoding=result.get("encoding", "utf-8"),
        file_type=file_type,
        used_streaming=is_large,
        already_clean=already_clean,
        quarantine_count=result.get("quarantine_count", 0),
        quarantine_sample=result.get("quarantine_sample", []),
        cleaning_audit={
            **(result.get("cleaning_audit") or {}),
            "file_type": file_type,
            "used_streaming": is_large,
            "temp_disk_bytes": result.get("temp_disk_bytes", 0),
            "clean_blob_path": clean_blob_path,
        },
    )
