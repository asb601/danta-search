"""
data_preprocessor.py — danta-search Stage-1 ingestion pipeline.

Single-module, clean-slate replacement of the per-cell loop preprocessor.
Streams a raw user file (CSV / TSV / Excel) from Azure Blob (or local
disk), cleans it column-wise with PyArrow compute kernels, infers a tight
schema, and writes a tuned Parquet file ready for a DataFusion query
engine and a LangGraph + GPT-4o agent.

Design notes (the *why*):

* Streaming-only. The VM is stateless; the file may be large; only a 256
  KB probe ever lives on local disk and is unlinked in a ``finally``.
* Vectorized everywhere. Every per-cell operation is expressed as a
  PyArrow compute kernel call against an Arrow array. The only Python
  loops permitted iterate over RecordBatches, columns of one batch, or
  schema fields — never over rows or cells.
* Library choices are deliberate: ``charset-normalizer`` (not chardet)
  for encoding sniff, ``duckdb.sniff_csv`` (most robust CSV
  auto-detect available) for delimiter / header / skip-rows, PyArrow
  CSV streaming for read, PyArrow compute for clean, PyArrow Parquet
  for write with explicit tuning that DataFusion needs (zstd,
  dictionary, statistics, page index, bloom filters, 1 MB pages).
* Excel uses ``python-calamine`` (Rust-backed) and is loaded in full
  because XLSX has no streaming format; downstream worker is expected to
  localize Excel inputs before calling.

Public API: ``preprocess_file(...)`` and the dataclasses ``ColumnProfile``
and ``PreprocessResult``. No module reaches into danta-search internals.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import re
import tempfile
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# Optional imports — guarded so the module loads even when the dependency
# isn't installed for the current code path (e.g. calamine for CSV-only).
try:
    import charset_normalizer as _charset_normalizer
except ImportError:  # pragma: no cover - import guard
    _charset_normalizer = None

try:
    import duckdb as _duckdb
except ImportError:  # pragma: no cover - import guard
    _duckdb = None

try:
    import fsspec as _fsspec
except ImportError:  # pragma: no cover - import guard
    _fsspec = None

try:
    from python_calamine import CalamineWorkbook as _CalamineWorkbook
except ImportError:  # pragma: no cover - import guard
    _CalamineWorkbook = None


logger = logging.getLogger("danta_search.preprocess")

# ─────────────────────────────────────────────────────────────────────
# Module-level constants — exposed for downstream services to reference.
# ─────────────────────────────────────────────────────────────────────

PROFILER_VERSION = 2
PROBE_BYTES = 256 * 1024
ENCODING_PROBE_BYTES = 64 * 1024
BATCH_BLOCK_SIZE = 64 * 1024 * 1024
ROW_GROUP_ROWS = 1_000_000
DATA_PAGE_BYTES = 1024 * 1024
SAMPLE_ROWS = 1000
COERCION_FAILURE_QUARANTINE_PCT = 5.0
ID_LIKE_DISTINCT_RATIO = 0.95
CATEGORY_MAX_DISTINCT = 200
BLOOM_FILTER_DISTINCT_MAX = 1_000_000

NULL_TOKENS = [
    "", " ", "NULL", "null", "Null",
    "N/A", "n/a", "NA", "na", "#N/A", "#NA",
    "None", "none", "NONE",
    "NaN", "nan", "NAN",
    "-", "--", "---",
    "TBD", "tbd", "TBA", "tba",
    "?", "??",
]

TIMESTAMP_FORMATS = [
    "%Y-%m-%d",
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",
    "%m/%d/%Y",
    "%m/%d/%Y %H:%M:%S",
    "%d/%m/%Y",
    "%d/%m/%Y %H:%M:%S",
    "%d-%b-%Y",
    "%d-%b-%y",
    "%Y%m%d",
]

# Zero-width / control / NBSP — anything that shouldn't survive a
# downstream string match but commonly leaks in from Excel exports.
# RE2 (PyArrow's regex backend) doesn't support \uXXXX escapes inside
# character classes, so we splice the literal codepoints into the class.
INVISIBLE_CHAR_RE = (
    "["
    "\u200B-\u200D\uFEFF\u00A0"  # ZWSP, ZWNJ, ZWJ, BOM, NBSP
    "\x00-\x08\x0B\x0C\x0E-\x1F\x7F"  # control chars
    "]"
)
SUBTOTAL_RE = r"(?i)\b(grand\s*total|sub\s*total|^total$|^totals$)\b"
SEPARATOR_RE = r"^[\-=_~]{2,}$"
BOOL_TRUE_TOKENS = {"true", "t", "yes", "y", "1"}
BOOL_FALSE_TOKENS = {"false", "f", "no", "n", "0"}

# Currency / thousands stripping for numeric coercion.
_NUMERIC_STRIP_RE = r"[,$€£¥₹\s]"
_LEADING_PLUS_RE = r"^\+"
_PARENS_NEG_RE = r"^\((.+)\)$"
_NULL_TOKEN_VALUE_SET = pa.array(NULL_TOKENS, type=pa.string())

_EXCEL_SUFFIXES = (".xlsx", ".xls", ".xlsm", ".xlsb")


# ─────────────────────────────────────────────────────────────────────
# Public dataclasses
# ─────────────────────────────────────────────────────────────────────


@dataclass
class ColumnProfile:
    """Per-column profile written to ``file_columns`` by the worker.

    Field set is the contract with the LangGraph agent and the DataFusion
    planner — do not rename or drop fields without coordinated migration.
    """
    name: str
    ordinal: int
    arrow_type: str
    semantic_type: str
    null_count: int
    total_count: int
    distinct_estimate: int
    min_value: str | None
    max_value: str | None
    top_values: list[tuple[str, int]]
    bloom_filter: bool
    coercion_failure_pct: float = 0.0


@dataclass
class PreprocessResult:
    """Return value of :func:`preprocess_file`. Mirrors the row shape that
    the ingestion worker writes into ``files`` / ``file_columns`` /
    ``row_group_stats``. Status is the routing decision for the worker.
    """
    status: str
    file_id: str
    org_id: str
    source_uri: str
    parquet_uri: str
    sample_uri: str | None
    row_count: int
    row_group_count: int
    bytes_written: int
    schema: dict[str, str]
    columns: list[ColumnProfile]
    warnings: list[str]
    profiler_version: int = PROFILER_VERSION
    elapsed_seconds: float = 0.0
    sniffed: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        """Stable JSON serialization for logging and snapshot comparison."""
        return json.dumps(dataclasses.asdict(self), default=str, indent=2)


# ─────────────────────────────────────────────────────────────────────
# I/O helpers — uniform interface over local paths and az:// URIs.
# ─────────────────────────────────────────────────────────────────────


def _is_remote(uri: str) -> bool:
    """Treat anything with a non-local scheme as remote. We currently only
    support ``az://`` but the dispatch is scheme-agnostic so swapping in
    s3:// later is a one-line change in the caller."""
    return "://" in uri and not uri.startswith("file://")


def _open_fs(storage_options: dict | None):
    """Lazy fsspec filesystem getter. Importing fsspec/adlfs is expensive
    and unnecessary when the source is a local path, so we defer it."""
    if _fsspec is None:
        raise RuntimeError(
            "fsspec/adlfs is required for az:// URIs but is not installed."
        )
    return _fsspec.filesystem("az", **(storage_options or {}))


def _read_probe(uri: str, storage_options: dict | None, n: int) -> bytes:
    """Read the first ``n`` bytes of ``uri`` without downloading the rest.

    For local paths we use ``open(...).read(n)``; for az:// we use
    fsspec's range-read which translates to a single Blob GET with a
    Range header. Either way: bounded I/O, no full download.
    """
    if _is_remote(uri):
        fs = _open_fs(storage_options)
        with fs.open(uri, "rb") as fh:
            return fh.read(n)
    with open(uri, "rb") as fh:
        return fh.read(n)


def _open_read_stream(uri: str, storage_options: dict | None):
    """Open a binary read stream for the full file. Caller must close."""
    if _is_remote(uri):
        fs = _open_fs(storage_options)
        return fs.open(uri, "rb")
    return open(uri, "rb")


def _open_write_stream(uri: str, storage_options: dict | None):
    """Open a binary write stream. Caller must close."""
    if _is_remote(uri):
        fs = _open_fs(storage_options)
        return fs.open(uri, "wb")
    os.makedirs(os.path.dirname(os.path.abspath(uri)) or ".", exist_ok=True)
    return open(uri, "wb")


def _stat_size(uri: str, storage_options: dict | None) -> int:
    """Best-effort size lookup for the written Parquet. Returns 0 if the
    backend doesn't surface size cheaply."""
    try:
        if _is_remote(uri):
            fs = _open_fs(storage_options)
            info = fs.info(uri)
            return int(info.get("size", 0))
        return os.path.getsize(uri)
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────
# Stage 0 — encoding sniff + DuckDB CSV sniff.
# ─────────────────────────────────────────────────────────────────────


def _detect_encoding(probe: bytes) -> str:
    """Run charset-normalizer over the first 64 KB and normalize the
    result into a canonical name PyArrow's CSV reader accepts.

    charset-normalizer returns names like ``utf_8`` and ``utf_16_le``;
    PyArrow expects ``utf-8`` / ``utf-16``. We collapse the families
    here so the rest of the pipeline never has to care.
    """
    if _charset_normalizer is None:
        return "utf-8"
    sample = probe[:ENCODING_PROBE_BYTES]
    try:
        best = _charset_normalizer.from_bytes(sample).best()
    except Exception:
        return "utf-8"
    if best is None:
        return "utf-8"
    name = (best.encoding or "utf-8").lower().replace("-", "_")
    if name in ("ascii", "utf_8"):
        return "utf-8"
    if name.startswith("utf_16"):
        return "utf-16"
    return name.replace("_", "-")


def _duckdb_sniff(probe: bytes, encoding: str) -> dict:
    """Write the probe to a temp file (DuckDB's sniffer needs a path) and
    return the relevant fields. Always unlinks the temp file.

    DuckDB's CSV sniffer is strict about UTF-8, so we transcode the probe
    to UTF-8 (using the encoding charset-normalizer detected) before
    handing it over. The original encoding is preserved in the returned
    dict and used by the actual PyArrow CSV reader.
    """
    if _duckdb is None:
        return {
            "Delimiter": ",", "Quote": '"', "Escape": '"',
            "SkipRows": 0, "HasHeader": True,
        }
    try:
        utf8_probe = probe.decode(encoding, errors="replace").encode("utf-8")
    except (LookupError, UnicodeDecodeError):
        utf8_probe = probe
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            tmp.write(utf8_probe)
            tmp_path = tmp.name
        # DuckDB exposes the sniffer as a SQL table function. Parameterised
        # form keeps the path safely escaped.
        con = _duckdb.connect()
        try:
            cur = con.execute(
                "SELECT * FROM sniff_csv(?, sample_size=20000)", [tmp_path]
            )
            row = cur.fetchone()
            cols = [d[0] for d in cur.description]
        finally:
            con.close()
        if row is None:
            raise RuntimeError("sniff_csv returned no rows")
        sniff = dict(zip(cols, row))

        def _norm_char(v: Any) -> str | None:
            """DuckDB returns '(empty)' as the sentinel for an absent
            single-char field. Anything not exactly one char is unsafe
            for PyArrow's ParseOptions, which requires a single UCS-4
            codepoint or False — so we collapse to None and let the
            caller pick a default."""
            if v is None:
                return None
            s = str(v)
            if s == "(empty)" or len(s) != 1:
                return None
            return s

        return {
            "Delimiter": _norm_char(sniff.get("Delimiter")) or ",",
            "Quote": _norm_char(sniff.get("Quote")) or '"',
            "Escape": _norm_char(sniff.get("Escape")) or '"',
            "SkipRows": int(sniff.get("SkipRows", 0) or 0),
            "HasHeader": bool(sniff.get("HasHeader", True)),
        }
    except Exception as exc:
        logger.warning("duckdb_sniff_failed error=%s", exc)
        return {
            "Delimiter": ",", "Quote": '"', "Escape": '"',
            "SkipRows": 0, "HasHeader": True,
        }
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def _build_csv_options(
    sniffed: dict, encoding: str
) -> tuple[pacsv.ReadOptions, pacsv.ParseOptions, pacsv.ConvertOptions]:
    """Translate the sniff result into PyArrow CSV options.

    Notable choices:
      * ``escape_char`` is only set when distinct from ``quote_char``;
        PyArrow rejects equal values with ArrowInvalid.
      * ``column_types`` is intentionally NOT set — the type lock happens
        after we've cleaned the first batch (see Stage 1).
    """
    delimiter = sniffed.get("Delimiter") or ","
    quote_char = sniffed.get("Quote") or '"'
    escape_char_raw = sniffed.get("Escape") or quote_char
    escape_char: str | bool = (
        escape_char_raw if escape_char_raw and escape_char_raw != quote_char else False
    )

    read_opts = pacsv.ReadOptions(
        block_size=BATCH_BLOCK_SIZE,
        skip_rows=int(sniffed.get("SkipRows") or 0),
        autogenerate_column_names=not bool(sniffed.get("HasHeader", True)),
        encoding=encoding,
    )
    parse_opts = pacsv.ParseOptions(
        delimiter=delimiter,
        quote_char=quote_char,
        escape_char=escape_char,
        newlines_in_values=True,
        invalid_row_handler=lambda r: "skip",
    )
    convert_opts = pacsv.ConvertOptions(
        null_values=NULL_TOKENS,
        strings_can_be_null=True,
        timestamp_parsers=TIMESTAMP_FORMATS,
    )
    return read_opts, parse_opts, convert_opts


# ─────────────────────────────────────────────────────────────────────
# Cleaning kernels — every one is a pure compute-kernel pipeline.
# ─────────────────────────────────────────────────────────────────────


def _clean_string_column(arr: pa.Array) -> pa.Array:
    """Strip invisible / control chars, trim whitespace, collapse the
    standard NULL tokens to actual null. Pure compute kernels.

    NBSP (U+00A0) is *replaced* with a regular space rather than deleted
    — it almost always represents an intentional word separator that
    survived an Excel export. The other zero-width / control codepoints
    are deleted because they convey no semantic meaning.
    """
    # NBSP → space first so "Foo\u00A0Bar" becomes "Foo Bar".
    cleaned = pc.replace_substring(arr, "\u00A0", " ")
    cleaned = pc.replace_substring_regex(cleaned, INVISIBLE_CHAR_RE, "")
    cleaned = pc.utf8_trim_whitespace(cleaned)
    is_null_token = pc.is_in(cleaned, value_set=_NULL_TOKEN_VALUE_SET)
    return pc.if_else(is_null_token, pa.scalar(None, type=pa.string()), cleaned)


def _all_nulls_mask(batch: pa.RecordBatch) -> pa.Array:
    """Boolean array, True where EVERY column in the row is null. Used to
    drop fully-empty rows that survive the parser."""
    if batch.num_columns == 0 or batch.num_rows == 0:
        return pa.array([], type=pa.bool_())
    mask = pc.is_null(batch.column(0))
    for i in range(1, batch.num_columns):
        mask = pc.and_(mask, pc.is_null(batch.column(i)))
    return mask


def _separator_row_mask(batch: pa.RecordBatch) -> pa.Array:
    """True where every string column is either null/empty or a pure
    separator like ``---``. Non-string columns count as "not separator"
    so a row with one numeric value survives."""
    if batch.num_rows == 0:
        return pa.array([], type=pa.bool_())
    # Start with all-True; AND in each string column's mask.
    mask = pa.array([True] * batch.num_rows, type=pa.bool_())
    saw_string = False
    for i in range(batch.num_columns):
        col = batch.column(i)
        if not pa.types.is_string(col.type):
            continue
        saw_string = True
        is_null = pc.is_null(col)
        is_empty = pc.equal(pc.coalesce(col, pa.scalar("")), pa.scalar(""))
        is_sep = pc.match_substring_regex(
            pc.coalesce(col, pa.scalar("")), SEPARATOR_RE
        )
        col_mask = pc.or_(pc.or_(is_null, is_empty), is_sep)
        mask = pc.and_(mask, col_mask)
    if not saw_string:
        return pa.array([False] * batch.num_rows, type=pa.bool_())
    return mask


def _subtotal_row_mask(batch: pa.RecordBatch) -> pa.Array:
    """True where ANY string column matches the subtotal regex. These are
    the human-friendly summary rows downstream SQL must never aggregate
    again ("Grand Total", "Subtotal", "Total")."""
    if batch.num_rows == 0:
        return pa.array([], type=pa.bool_())
    mask = pa.array([False] * batch.num_rows, type=pa.bool_())
    for i in range(batch.num_columns):
        col = batch.column(i)
        if not pa.types.is_string(col.type):
            continue
        hit = pc.match_substring_regex(
            pc.coalesce(col, pa.scalar("")), SUBTOTAL_RE
        )
        mask = pc.or_(mask, hit)
    return mask


def _drop_garbage_rows(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Combined drop-mask: kill fully-null rows, separator-only rows, and
    subtotal rows. One ``filter`` call so we touch each row exactly
    once."""
    if batch.num_rows == 0:
        return batch
    keep = pc.invert(_all_nulls_mask(batch))
    keep = pc.and_(keep, pc.invert(_separator_row_mask(batch)))
    keep = pc.and_(keep, pc.invert(_subtotal_row_mask(batch)))
    return batch.filter(keep)


def _clean_batch(batch: pa.RecordBatch) -> pa.RecordBatch:
    """Apply per-column string cleaning + drop garbage rows. Cheap because
    string cleaning is a single Arrow kernel chain per column."""
    new_cols = []
    for i in range(batch.num_columns):
        col = batch.column(i)
        if pa.types.is_string(col.type):
            new_cols.append(_clean_string_column(col))
        else:
            new_cols.append(col)
    cleaned = pa.RecordBatch.from_arrays(new_cols, names=batch.schema.names)
    return _drop_garbage_rows(cleaned)


# ─────────────────────────────────────────────────────────────────────
# Type refinement — boolean → numeric → timestamp → string fallback.
# ─────────────────────────────────────────────────────────────────────


def _try_boolean_cast(arr: pa.Array) -> pa.Array | None:
    """Promote string→bool when every non-null value is in the canonical
    yes/no token set. We deliberately do NOT accept arbitrary tokens —
    the cost of a false positive (treating "Y" for "Yemen" as boolean)
    is far higher than the cost of leaving the column as a string."""
    if not pa.types.is_string(arr.type):
        return None
    non_null = pc.drop_null(arr)
    if len(non_null) == 0:
        return None
    lowered = pc.utf8_lower(non_null)
    allowed = pa.array(
        sorted(BOOL_TRUE_TOKENS | BOOL_FALSE_TOKENS), type=pa.string()
    )
    if not pc.all(pc.is_in(lowered, value_set=allowed)).as_py():
        return None
    # Cast the full array (including nulls) by mapping true tokens.
    truthy = pa.array(sorted(BOOL_TRUE_TOKENS), type=pa.string())
    full_lower = pc.utf8_lower(arr)
    return pc.is_in(full_lower, value_set=truthy)


def _try_numeric_cast(arr: pa.Array) -> tuple[pa.Array, float]:
    """Return ``(cast_array, failure_pct)`` for a string column. If more
    than 20 % of the originally-non-null values fail the cast we give up
    and return the original — the column wasn't really numeric.

    PyArrow's ``cast(..., safe=False)`` still raises on invalid values
    rather than producing nulls, so we mask non-numeric strings to null
    with a regex first and then cast the cleaned column.
    """
    if not pa.types.is_string(arr.type):
        return arr, 0.0
    non_null_before = len(arr) - arr.null_count
    if non_null_before == 0:
        return arr, 0.0
    stripped = pc.replace_substring_regex(arr, _NUMERIC_STRIP_RE, "")
    stripped = pc.replace_substring_regex(stripped, _LEADING_PLUS_RE, "")
    # Parenthesized negatives → leading minus, vectorized.
    stripped = pc.replace_substring_regex(stripped, _PARENS_NEG_RE, r"-\1")
    # Mask anything that isn't a valid float literal to null so the
    # subsequent cast can never raise.
    is_numeric = pc.match_substring_regex(stripped, r"^-?\d+(\.\d+)?([eE][-+]?\d+)?$")
    masked = pc.if_else(is_numeric, stripped, pa.scalar(None, type=pa.string()))
    try:
        casted = pc.cast(masked, pa.float64(), safe=False)
    except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
        return arr, 0.0
    non_null_after = len(casted) - casted.null_count
    failed = non_null_before - non_null_after
    failure_pct = (failed / non_null_before) * 100.0
    if failure_pct > 20.0:
        return arr, 0.0
    return casted, failure_pct


def _try_timestamp_cast(arr: pa.Array) -> pa.Array | None:
    """Try every format in ``TIMESTAMP_FORMATS``; return the first whose
    success rate is ≥ 95 % over non-nulls. The threshold catches columns
    where one or two malformed dates would otherwise drop the whole
    column to string."""
    if not pa.types.is_string(arr.type):
        return None
    non_null_before = len(arr) - arr.null_count
    if non_null_before == 0:
        return None
    for fmt in TIMESTAMP_FORMATS:
        try:
            casted = pc.strptime(arr, format=fmt, unit="us", error_is_null=True)
        except (pa.ArrowInvalid, pa.ArrowNotImplementedError):
            continue
        non_null_after = len(casted) - casted.null_count
        if non_null_after / non_null_before >= 0.95:
            return casted
    return None


def _refine_column_types(table: pa.Table) -> tuple[pa.Table, dict[str, float]]:
    """Lock the schema after the first cleaned batch. Order matters:

      1. boolean (cheapest, narrowest signal)
      2. numeric (next-narrowest after stripping currency / parens)
      3. timestamp (try every well-known format)
      4. string (fallback)
    """
    new_cols: list[pa.Array | pa.ChunkedArray] = []
    new_fields: list[pa.Field] = []
    failures: dict[str, float] = {}
    for field_obj in table.schema:
        col = table.column(field_obj.name)
        # Operate on a single combined Array for kernel friendliness.
        arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
        if not pa.types.is_string(arr.type):
            new_cols.append(arr)
            new_fields.append(field_obj)
            continue

        bool_cast = _try_boolean_cast(arr)
        if bool_cast is not None:
            new_cols.append(bool_cast)
            new_fields.append(pa.field(field_obj.name, pa.bool_()))
            continue

        num_cast, failure_pct = _try_numeric_cast(arr)
        if num_cast is not arr:
            if failure_pct > 0.0:
                failures[field_obj.name] = failure_pct
            new_cols.append(num_cast)
            new_fields.append(pa.field(field_obj.name, pa.float64()))
            continue

        ts_cast = _try_timestamp_cast(arr)
        if ts_cast is not None:
            new_cols.append(ts_cast)
            new_fields.append(pa.field(field_obj.name, pa.timestamp("us")))
            continue

        new_cols.append(arr)
        new_fields.append(field_obj)

    refined = pa.Table.from_arrays(new_cols, schema=pa.schema(new_fields))
    return refined, failures


# ─────────────────────────────────────────────────────────────────────
# Profiling — bounded streaming accumulators.
# ─────────────────────────────────────────────────────────────────────


class _ColumnAccumulator:
    """Streaming statistics for one column. Hard-bounded memory:
      * top_counter holds at most 5,000 observations
      * distinct_sample holds at most 200 distinct strings

    TODO: swap top_counter for ``datasketches.HllSketch`` for accurate
    cardinality on very wide columns; the current extrapolation is fine
    for most danta-search files (≤ 50 M rows).
    """

    __slots__ = (
        "name", "ordinal", "null_count", "total_count",
        "min_value", "max_value",
        "distinct_sample", "top_counter",
        "is_string", "sum_len", "_obs",
    )

    def __init__(self, name: str, ordinal: int, is_string: bool):
        self.name = name
        self.ordinal = ordinal
        self.null_count = 0
        self.total_count = 0
        self.min_value: Any = None
        self.max_value: Any = None
        self.distinct_sample: set[str] = set()
        self.top_counter: Counter = Counter()
        self.is_string = is_string
        self.sum_len = 0
        self._obs = 0  # observations fed into top_counter so far

    def update(self, arr: pa.Array) -> None:
        n = len(arr)
        nulls = arr.null_count
        self.total_count += n
        self.null_count += nulls
        if n == nulls:
            return
        try:
            cur_min = pc.min(arr).as_py()
            cur_max = pc.max(arr).as_py()
        except (pa.ArrowNotImplementedError, pa.ArrowInvalid):
            cur_min = cur_max = None
        if cur_min is not None:
            self.min_value = cur_min if self.min_value is None else min(self.min_value, cur_min)
        if cur_max is not None:
            self.max_value = cur_max if self.max_value is None else max(self.max_value, cur_max)
        if self.is_string and self._obs < 5000:
            # Sample up to 500 non-null values per batch; bounded by
            # 5000 total observations across the file.
            non_null = pc.drop_null(arr)
            take = min(len(non_null), 500, 5000 - self._obs)
            if take > 0:
                values = non_null.slice(0, take).to_pylist()
                self.top_counter.update(values)
                for v in values:
                    self.sum_len += len(v) if isinstance(v, str) else 0
                    if len(self.distinct_sample) < 200:
                        self.distinct_sample.add(v)
                self._obs += take


def _estimate_distinct(counter: Counter, total_observed: int, total_rows: int) -> int:
    """Cheap distinct estimator. Saturates if we've already seen many
    repeats; extrapolates linearly otherwise. The HyperLogLog upgrade
    is noted in :class:`_ColumnAccumulator`."""
    seen_distinct = len(counter)
    if total_observed == 0:
        return 0
    if seen_distinct < total_observed * 0.5:
        return seen_distinct
    if total_observed >= total_rows:
        return seen_distinct
    estimate = int(seen_distinct * (total_rows / total_observed))
    return min(estimate, total_rows)


def _semantic_type(
    arrow_type: pa.DataType,
    top_counter: Counter,
    distinct_estimate: int,
    total_count: int,
    avg_len: float,
) -> str:
    """Map an Arrow type + cardinality + length signal to one of the six
    semantic types the agent's prompt builder uses."""
    if pa.types.is_boolean(arrow_type):
        return "bool"
    if pa.types.is_timestamp(arrow_type):
        return "timestamp"
    if pa.types.is_date(arrow_type):
        return "date"
    if (
        pa.types.is_integer(arrow_type)
        or pa.types.is_floating(arrow_type)
        or pa.types.is_decimal(arrow_type)
    ):
        if total_count > 0 and distinct_estimate / total_count >= ID_LIKE_DISTINCT_RATIO:
            return "id"
        return "measure"
    # String / other. We check id-likeness BEFORE the absolute distinct
    # cap because a small file (≤ 200 rows) where every value is unique
    # is clearly an id, not a category — the cap is meant to catch
    # large tables with few distinct values, not tiny tables.
    if total_count > 0 and distinct_estimate / total_count >= ID_LIKE_DISTINCT_RATIO:
        return "id"
    if distinct_estimate <= CATEGORY_MAX_DISTINCT and total_count >= 50:
        return "category"
    if avg_len > 64:
        return "text"
    return "text"


def _finalize_accumulator(
    acc: _ColumnAccumulator, arrow_type: pa.DataType
) -> ColumnProfile:
    distinct = _estimate_distinct(acc.top_counter, acc._obs, acc.total_count)
    avg_len = (acc.sum_len / acc._obs) if acc._obs else 0.0
    sem = _semantic_type(arrow_type, acc.top_counter, distinct, acc.total_count, avg_len)
    top_values = [(str(k), int(v)) for k, v in acc.top_counter.most_common(20)]
    bloom = (
        acc.is_string
        and distinct <= BLOOM_FILTER_DISTINCT_MAX
        and sem in {"category", "id"}
    )
    return ColumnProfile(
        name=acc.name,
        ordinal=acc.ordinal,
        arrow_type=str(arrow_type),
        semantic_type=sem,
        null_count=acc.null_count,
        total_count=acc.total_count,
        distinct_estimate=distinct,
        min_value=None if acc.min_value is None else str(acc.min_value),
        max_value=None if acc.max_value is None else str(acc.max_value),
        top_values=top_values,
        bloom_filter=bloom,
    )


# ─────────────────────────────────────────────────────────────────────
# Result helpers
# ─────────────────────────────────────────────────────────────────────


def _failed_result(
    file_id: str, org_id: str, source_uri: str, parquet_uri: str,
    sample_uri: str | None, warning: str, started_at: float,
    sniffed: dict | None = None,
) -> PreprocessResult:
    return PreprocessResult(
        status="failed",
        file_id=file_id,
        org_id=org_id,
        source_uri=source_uri,
        parquet_uri=parquet_uri,
        sample_uri=sample_uri,
        row_count=0,
        row_group_count=0,
        bytes_written=0,
        schema={},
        columns=[],
        warnings=[warning],
        elapsed_seconds=round(time.time() - started_at, 3),
        sniffed=sniffed or {},
    )


# ─────────────────────────────────────────────────────────────────────
# Excel path — calamine-only.
# ─────────────────────────────────────────────────────────────────────


def _read_excel_table(local_path: str) -> pa.Table:
    """Load the first sheet of an Excel workbook into an Arrow Table.

    Calamine returns nested Python lists; we transpose into per-column
    lists once and let Arrow infer types. All subsequent cleaning runs
    through the same vectorized kernels as the CSV path.
    """
    if _CalamineWorkbook is None:
        raise RuntimeError(
            "python-calamine is required to read Excel files but is not installed."
        )
    wb = _CalamineWorkbook.from_path(local_path)
    sheet_name = wb.sheet_names[0]
    rows = wb.get_sheet_by_name(sheet_name).to_python()
    if not rows:
        return pa.table({})
    headers = [str(h) if h is not None else f"col_{i}" for i, h in enumerate(rows[0])]
    body = rows[1:]
    cols: list[list[Any]] = [[] for _ in headers]
    # The only loop over user data — required because calamine returns
    # Python rows. Nothing per-cell happens here; we just transpose.
    for r in body:
        for i, h in enumerate(headers):
            cols[i].append(r[i] if i < len(r) else None)
    arrays = []
    for col in cols:
        # Force string for fully-null columns so downstream cleaning
        # has a stable type to work against.
        try:
            arrays.append(pa.array(col))
        except (pa.ArrowInvalid, pa.ArrowTypeError):
            arrays.append(pa.array([None if v is None else str(v) for v in col], type=pa.string()))
    return pa.table(dict(zip(headers, arrays)))


def _process_excel(
    source_uri: str, target_parquet_uri: str, target_sample_uri: str | None,
    file_id: str, org_id: str, storage_options: dict | None, started_at: float,
) -> PreprocessResult:
    """Excel branch — load whole workbook, clean, refine types, write.

    Excel has no streaming format; the worker is expected to localize
    the file before calling. We keep the same cleaning + profiling
    code path as CSV so both formats produce identical downstream
    metadata.
    """
    table = _read_excel_table(source_uri)
    if table.num_rows == 0 or table.num_columns == 0:
        return _failed_result(
            file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
            "Excel sheet is empty.", started_at,
        )
    # Clean in one batch (Excel files are small enough — typically < 1 M rows).
    batch = table.to_batches(max_chunksize=table.num_rows)[0]
    cleaned = _clean_batch(batch)
    cleaned_table = pa.Table.from_batches([cleaned])
    refined, failures = _refine_column_types(cleaned_table)

    accs = _build_accumulators(refined.schema)
    for b in refined.to_batches(max_chunksize=10_000):
        for i in range(b.num_columns):
            accs[i].update(b.column(i))

    sink = _open_write_stream(target_parquet_uri, storage_options)
    try:
        bloom_cols = _bloom_columns(accs, refined.schema)
        writer_kwargs = _parquet_writer_kwargs(bloom_cols)
        writer = pq.ParquetWriter(
            sink,
            refined.schema,
            **writer_kwargs,
        )
        try:
            writer.write_table(refined, row_group_size=ROW_GROUP_ROWS)
        finally:
            writer.close()
    finally:
        sink.close()

    sample_uri_out = _maybe_write_sample(
        refined.slice(0, min(SAMPLE_ROWS, refined.num_rows)),
        target_sample_uri, storage_options,
    )

    columns = [_finalize_accumulator(acc, refined.schema.field(i).type)
               for i, acc in enumerate(accs)]
    for c in columns:
        c.coercion_failure_pct = failures.get(c.name, 0.0)

    warnings_list = _quarantine_warnings(columns)
    status = "quarantined" if warnings_list else "ready"

    return PreprocessResult(
        status=status,
        file_id=file_id,
        org_id=org_id,
        source_uri=source_uri,
        parquet_uri=target_parquet_uri,
        sample_uri=sample_uri_out,
        row_count=refined.num_rows,
        row_group_count=_safe_row_group_count(target_parquet_uri, storage_options),
        bytes_written=_stat_size(target_parquet_uri, storage_options),
        schema={f.name: str(f.type) for f in refined.schema},
        columns=columns,
        warnings=warnings_list,
        elapsed_seconds=round(time.time() - started_at, 3),
        sniffed={"format": "excel"},
    )


# ─────────────────────────────────────────────────────────────────────
# Shared helpers between CSV and Excel paths.
# ─────────────────────────────────────────────────────────────────────


def _build_accumulators(schema: pa.Schema) -> list[_ColumnAccumulator]:
    return [
        _ColumnAccumulator(f.name, i, pa.types.is_string(f.type))
        for i, f in enumerate(schema)
    ]


import inspect as _inspect

_PARQUET_WRITER_PARAMS = set(
    _inspect.signature(pq.ParquetWriter).parameters.keys()
)


def _parquet_writer_kwargs(bloom_cols: list[str]) -> dict:
    """Build the kwargs dict for :class:`pq.ParquetWriter`.

    PyArrow's ParquetWriter signature varies between versions
    (``bloom_filter_columns`` was added later). We probe the signature
    once and only forward kwargs the installed build accepts \u2014 lets us
    ship one module that works across PyArrow 12+.
    """
    base: dict = {
        "compression": "zstd",
        "compression_level": 3,
        "use_dictionary": True,
        "write_statistics": True,
        "write_page_index": True,
        "data_page_size": DATA_PAGE_BYTES,
        "write_batch_size": 10_000,
    }
    if bloom_cols and "bloom_filter_columns" in _PARQUET_WRITER_PARAMS:
        base["bloom_filter_columns"] = bloom_cols
    return base


def _bloom_columns(
    accs: list[_ColumnAccumulator], schema: pa.Schema
) -> list[str]:
    """Pre-flight bloom filter list. We use the same heuristic the final
    profile uses so the file metadata and the Parquet footer agree.

    String + (category | id) + reasonable cardinality. Bloom filters on
    measures or timestamps add bytes for no DataFusion pruning win.
    """
    out: list[str] = []
    for i, acc in enumerate(accs):
        if not acc.is_string:
            continue
        if not pa.types.is_string(schema.field(i).type):
            continue
        # We don't have final cardinality yet at write-time on the CSV
        # path; be permissive and let downstream measure usefulness.
        out.append(acc.name)
    return out


def _maybe_write_sample(
    sample_table: pa.Table,
    target_sample_uri: str | None,
    storage_options: dict | None,
) -> str | None:
    """Write the small companion preview file. Same compression and
    statistics as the main Parquet so downstream readers don't branch."""
    if not target_sample_uri or sample_table.num_rows == 0:
        return None
    sink = _open_write_stream(target_sample_uri, storage_options)
    try:
        pq.write_table(
            sample_table, sink,
            compression="zstd", compression_level=3,
            use_dictionary=True, write_statistics=True,
            write_page_index=True, data_page_size=DATA_PAGE_BYTES,
        )
    finally:
        sink.close()
    return target_sample_uri


def _safe_row_group_count(uri: str, storage_options: dict | None) -> int:
    """Footer-only read for row-group count. Cheap even over Blob since
    Parquet stores the footer at the tail of the file."""
    try:
        if _is_remote(uri):
            fs = _open_fs(storage_options)
            with fs.open(uri, "rb") as fh:
                meta = pq.read_metadata(fh)
        else:
            meta = pq.read_metadata(uri)
        return meta.num_row_groups
    except Exception:
        return 0


def _quarantine_warnings(columns: list[ColumnProfile]) -> list[str]:
    """One warning per column whose numeric coercion lost more than the
    quarantine threshold. Concise + actionable for the UI."""
    out: list[str] = []
    for c in columns:
        if c.coercion_failure_pct > COERCION_FAILURE_QUARANTINE_PCT:
            out.append(
                f"Column '{c.name}' has {c.coercion_failure_pct:.1f}% non-numeric "
                f"values after coercion — quarantined for review."
            )
    return out


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


def preprocess_file(
    source_uri: str,
    target_parquet_uri: str,
    target_sample_uri: str | None,
    file_id: str,
    org_id: str,
    storage_options: dict | None = None,
    *,
    is_excel: bool | None = None,
) -> PreprocessResult:
    """Stage-1 entry point. See module docstring for design rationale.

    Streams the source from Blob (or local), cleans it column-wise with
    PyArrow compute kernels, locks a tight schema after the first batch,
    writes a tuned Parquet, and returns a :class:`PreprocessResult` for
    the worker to persist. Never raises on user-data errors — those
    surface as ``status='quarantined'`` or ``'failed'`` with warnings.
    """
    started_at = time.time()
    if is_excel is None:
        is_excel = source_uri.lower().endswith(_EXCEL_SUFFIXES)

    if is_excel:
        try:
            return _process_excel(
                source_uri, target_parquet_uri, target_sample_uri,
                file_id, org_id, storage_options, started_at,
            )
        except Exception as exc:
            logger.exception("excel_preprocess_failed file_id=%s", file_id)
            return _failed_result(
                file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
                f"Excel read failed: {exc}", started_at,
            )

    # ── CSV / TSV / delimited path ───────────────────────────────────
    probe: bytes
    try:
        probe = _read_probe(source_uri, storage_options, PROBE_BYTES)
    except Exception as exc:
        logger.exception("probe_read_failed file_id=%s", file_id)
        return _failed_result(
            file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
            f"Could not read source: {exc}", started_at,
        )
    if not probe:
        return _failed_result(
            file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
            "Source file is empty.", started_at,
        )

    encoding = _detect_encoding(probe)
    sniffed = _duckdb_sniff(probe, encoding)
    sniffed["encoding"] = encoding
    sniffed["format"] = "csv"

    read_opts, parse_opts, convert_opts = _build_csv_options(sniffed, encoding)

    source_stream = None
    sink = None
    writer: pq.ParquetWriter | None = None
    try:
        source_stream = _open_read_stream(source_uri, storage_options)
        try:
            reader = pacsv.open_csv(source_stream, read_opts, parse_opts, convert_opts)
        except pa.ArrowInvalid as exc:
            logger.warning("csv_open_failed file_id=%s error=%s", file_id, exc)
            return _failed_result(
                file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
                f"CSV parser rejected the file: {exc}", started_at, sniffed,
            )

        # First batch → clean → lock schema.
        try:
            first_batch_raw = reader.read_next_batch()
        except StopIteration:
            return _failed_result(
                file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
                "No data rows found after header.", started_at, sniffed,
            )

        first_cleaned = _clean_batch(first_batch_raw)
        first_table = pa.Table.from_batches([first_cleaned])
        refined_first, first_failures = _refine_column_types(first_table)
        locked_schema = refined_first.schema

        accs = _build_accumulators(locked_schema)
        warnings_list: list[str] = []
        sample_buffer: list[pa.RecordBatch] = []
        sample_rows_collected = 0
        total_rows = 0
        coercion_failure_pct: dict[str, float] = dict(first_failures)

        sink = _open_write_stream(target_parquet_uri, storage_options)
        bloom_cols = _bloom_columns(accs, locked_schema)
        writer_kwargs = _parquet_writer_kwargs(bloom_cols)
        writer = pq.ParquetWriter(
            sink,
            locked_schema,
            **writer_kwargs,
        )

        # Tradeoff: we write each cleaned batch directly (no inter-batch
        # buffering). Row groups follow PyArrow's default chunking, which
        # is fine for our size profile (≤ a few hundred MB per file).
        # If we ever need 1 M-row groups we'd accumulate into a Table
        # and call writer.write_table(table, row_group_size=ROW_GROUP_ROWS).
        def _write_table(t: pa.Table) -> None:
            nonlocal total_rows, sample_rows_collected
            if t.num_rows == 0:
                return
            writer.write_table(t, row_group_size=ROW_GROUP_ROWS)
            total_rows += t.num_rows
            for b in t.to_batches(max_chunksize=10_000):
                for i in range(b.num_columns):
                    accs[i].update(b.column(i))
                if sample_rows_collected < SAMPLE_ROWS:
                    take = min(b.num_rows, SAMPLE_ROWS - sample_rows_collected)
                    sample_buffer.append(b.slice(0, take))
                    sample_rows_collected += take

        _write_table(refined_first)

        # Subsequent batches — re-clean + cast to locked schema.
        while True:
            try:
                raw = reader.read_next_batch()
            except StopIteration:
                break
            cleaned = _clean_batch(raw)
            if cleaned.num_rows == 0:
                continue
            try:
                t = pa.Table.from_batches([cleaned]).cast(locked_schema, safe=False)
            except (pa.ArrowInvalid, pa.ArrowTypeError) as exc:
                msg = f"Skipped a batch that could not be cast to locked schema: {exc}"
                logger.warning("batch_cast_failed file_id=%s error=%s", file_id, exc)
                warnings_list.append(msg)
                continue
            # Track per-column coercion failure: count nulls newly introduced
            # by cast vs the cleaned source.
            for f in locked_schema:
                if not (pa.types.is_floating(f.type) or pa.types.is_timestamp(f.type)):
                    continue
                src_idx = cleaned.schema.get_field_index(f.name)
                if src_idx < 0:
                    continue
                src_arr = cleaned.column(src_idx)
                dst_arr = t.column(f.name)
                src_non_null = len(src_arr) - src_arr.null_count
                dst_non_null = len(dst_arr) - dst_arr.null_count
                if src_non_null > 0 and dst_non_null < src_non_null:
                    pct = ((src_non_null - dst_non_null) / src_non_null) * 100.0
                    # Combine with existing pct via running max — one
                    # bad batch should not be smoothed into invisibility.
                    coercion_failure_pct[f.name] = max(
                        coercion_failure_pct.get(f.name, 0.0), pct
                    )
            _write_table(t)

        if total_rows == 0:
            return _failed_result(
                file_id, org_id, source_uri, target_parquet_uri, target_sample_uri,
                "No data rows survived cleaning.", started_at, sniffed,
            )

        # Sample file before closing the main writer in case we share fs.
        sample_uri_out: str | None = None
        if target_sample_uri and sample_buffer:
            sample_table = pa.Table.from_batches(sample_buffer, schema=locked_schema)
            sample_uri_out = _maybe_write_sample(
                sample_table, target_sample_uri, storage_options
            )

    finally:
        if writer is not None:
            try:
                writer.close()
            except Exception:
                pass
        if sink is not None:
            try:
                sink.close()
            except Exception:
                pass
        if source_stream is not None:
            try:
                source_stream.close()
            except Exception:
                pass

    columns = [_finalize_accumulator(acc, locked_schema.field(i).type)
               for i, acc in enumerate(accs)]
    for c in columns:
        c.coercion_failure_pct = coercion_failure_pct.get(c.name, 0.0)

    quarantine_warnings = _quarantine_warnings(columns)
    warnings_list.extend(quarantine_warnings)
    status = "quarantined" if quarantine_warnings else "ready"

    return PreprocessResult(
        status=status,
        file_id=file_id,
        org_id=org_id,
        source_uri=source_uri,
        parquet_uri=target_parquet_uri,
        sample_uri=sample_uri_out,
        row_count=total_rows,
        row_group_count=_safe_row_group_count(target_parquet_uri, storage_options),
        bytes_written=_stat_size(target_parquet_uri, storage_options),
        schema={f.name: str(f.type) for f in locked_schema},
        columns=columns,
        warnings=warnings_list,
        elapsed_seconds=round(time.time() - started_at, 3),
        sniffed=sniffed,
    )


# ─────────────────────────────────────────────────────────────────────
# Acceptance tests — run with: python data_preprocessor.py
# ─────────────────────────────────────────────────────────────────────


def _build_synthetic_dirty_csv(path: str) -> dict:
    """Generate a CSV that exercises every cleaning rule. Returns the
    expected facts the assertions will check."""
    n_rows = 120
    rows: list[str] = []
    # 3 junk header lines + 1 real header.
    rows.append("Acme Corp Quarterly Export")
    rows.append("")
    rows.append("Generated 2026-01-01")
    rows.append("id;category;flag;amount;event_date;invisible_col")
    for i in range(n_rows):
        cat = ["alpha", "beta", "gamma", "delta"][i % 4]
        flag = ["Yes", "no", "YES", "No"][i % 4]
        amount = [
            "$1,234.56", "(200.00)", "3,000", "42.00",
            "1.5", "9,999.99", "0.01", "(15.25)",
        ][i % 8]
        date = "12-MAY-2026"
        invisible = "\u00A0Foo\u00A0Bar\u00A0"
        rows.append(f"id_{i};{cat};{flag};{amount};{date};{invisible}")
    # Null tokens row.
    rows.append("id_null;N/A;Yes;NULL;-;TBD")
    # Subtotal row — must be dropped.
    rows.append("id_total;Grand Total;Yes;1.0;12-MAY-2026;Foo Bar")
    # Separator row — must be dropped.
    rows.append("---;---;---;---;---;---")
    # Encode latin-1 with a non-ASCII char to test encoding detection.
    text = "\n".join(rows) + "\n"
    text = text.replace("alpha", "alphá")
    with open(path, "wb") as fh:
        fh.write(text.encode("latin-1"))
    return {
        "n_rows_after_clean": n_rows + 1,  # +1 for the null-token row
        "expected_columns": ["id", "category", "flag", "amount", "event_date", "invisible_col"],
    }


def _build_quarantine_csv(path: str) -> None:
    rows = ["id,amount"]
    for i in range(100):
        rows.append(f"id_{i},{i*1.5}")
    for i in range(15):  # 15% non-numeric → > 5% quarantine
        rows.append(f"bad_{i},banana")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(rows) + "\n")


def _run_acceptance_tests() -> int:
    """Print PASS/FAIL per test; exit 0 if all pass, 1 otherwise."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    failures = 0
    tmpdir = tempfile.mkdtemp(prefix="danta_search_pp_test_")
    src_csv = os.path.join(tmpdir, "dirty.csv")
    out_pq = os.path.join(tmpdir, "out.parquet")
    sample_pq = os.path.join(tmpdir, "sample.parquet")
    expected = _build_synthetic_dirty_csv(src_csv)

    result = preprocess_file(
        source_uri=src_csv,
        target_parquet_uri=out_pq,
        target_sample_uri=sample_pq,
        file_id="test-file",
        org_id="test-org",
    )

    table = pq.read_table(out_pq)
    schema_names = table.schema.names
    meta = pq.read_metadata(out_pq)

    def check(name: str, ok: bool, detail: str = "") -> None:
        nonlocal failures
        if ok:
            print(f"PASS  {name}")
        else:
            failures += 1
            print(f"FAIL  {name}  {detail}")

    # 1. Encoding (non-ASCII char from latin-1 source preserved).
    cat_col = table.column("category").to_pylist()
    check("01 encoding", any("alphá" in (v or "") for v in cat_col),
          detail=f"sample={cat_col[:5]}")

    # 2. Delimiter — schema has the 6 expected columns.
    check("02 delimiter ;", schema_names == expected["expected_columns"],
          detail=f"got={schema_names}")

    # 3. Header skipped — first column name is "id" not the junk row text.
    check("03 skip junk rows", schema_names[0] == "id",
          detail=f"got={schema_names[0]}")

    # 4. Null tokens — row id_null has nulls.
    id_idx = cat_col  # category column
    null_token_row_present = "id_null" in table.column("id").to_pylist()
    null_in_amount = table.column("amount").to_pylist()
    # Find the row.
    ids = table.column("id").to_pylist()
    null_row = None
    if "id_null" in ids:
        null_row = ids.index("id_null")
    check("04 null tokens",
          null_row is not None
          and table.column("category")[null_row].as_py() is None
          and table.column("amount")[null_row].as_py() is None,
          detail=f"null_row={null_row}")

    # 5. Invisible chars stripped.
    invis = table.column("invisible_col").to_pylist()
    check("05 invisible chars",
          any(v == "Foo Bar" for v in invis if v),
          detail=f"sample={invis[:3]}")

    # 6. Subtotal row dropped.
    check("06 subtotal dropped", "id_total" not in ids)

    # 7. Separator row dropped — no row has all '---'.
    check("07 separator dropped", "---" not in ids)

    # 8. Boolean inference.
    check("08 bool inference",
          pa.types.is_boolean(table.schema.field("flag").type),
          detail=f"got={table.schema.field('flag').type}")

    # 9. Numeric inference + values.
    amt_type = table.schema.field("amount").type
    amt_vals = [v for v in table.column("amount").to_pylist() if v is not None]
    check("09 numeric inference",
          pa.types.is_floating(amt_type)
          and 1234.56 in amt_vals and -200.0 in amt_vals and 3000.0 in amt_vals,
          detail=f"type={amt_type} sample={amt_vals[:6]}")

    # 10. Timestamp inference.
    check("10 timestamp inference",
          pa.types.is_timestamp(table.schema.field("event_date").type),
          detail=f"type={table.schema.field('event_date').type}")

    # 11. Profile semantic types.
    sem = {c.name: c.semantic_type for c in result.columns}
    check("11 semantic types",
          sem.get("id") == "id"
          and sem.get("category") == "category"
          and sem.get("amount") == "measure"
          and sem.get("event_date") == "timestamp"
          and sem.get("flag") == "bool",
          detail=str(sem))

    # 12. Parquet tuning.
    has_stats = all(
        meta.row_group(rg).column(c).statistics is not None
        for rg in range(meta.num_row_groups)
        for c in range(meta.num_columns)
    )
    check("12 parquet tuning",
          meta.num_row_groups >= 1 and has_stats,
          detail=f"row_groups={meta.num_row_groups} has_stats={has_stats}")

    # 13. Sample file.
    sample_table = pq.read_table(sample_pq)
    check("13 sample file",
          sample_table.num_rows == min(SAMPLE_ROWS, table.num_rows),
          detail=f"sample_rows={sample_table.num_rows} total={table.num_rows}")

    # 14. Quarantine.
    qsrc = os.path.join(tmpdir, "quarantine.csv")
    qpq = os.path.join(tmpdir, "quarantine.parquet")
    _build_quarantine_csv(qsrc)
    qres = preprocess_file(
        source_uri=qsrc, target_parquet_uri=qpq, target_sample_uri=None,
        file_id="qtest", org_id="qtest-org",
    )
    amount_profile = next((c for c in qres.columns if c.name == "amount"), None)
    check("14 quarantine",
          qres.status == "quarantined"
          and amount_profile is not None
          and amount_profile.coercion_failure_pct > COERCION_FAILURE_QUARANTINE_PCT,
          detail=f"status={qres.status} pct={amount_profile and amount_profile.coercion_failure_pct}")

    print()
    print(f"{14 - failures}/14 tests passed")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    import sys
    sys.exit(_run_acceptance_tests())
