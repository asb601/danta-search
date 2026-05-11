"""
DataFusion query client — replaces duckdb_client.execute_query_sync.

Why this exists
---------------
DuckDB uses thread-local connections with shared internal structures (buffer
manager, file handles). Under concurrent load, 10 simultaneous queries
compete for those structures and effectively serialize — a 2s query becomes
20s when 9 others are queued.

DataFusion fixes this with per-request SessionContext: each query gets an
isolated execution environment with zero shared mutable state. 40 concurrent
queries run truly in parallel, bounded only by CPU and Azure Blob bandwidth.

SQL format: UNCHANGED from DuckDB.
  The AI still generates:   read_parquet('az://CONTAINER/file.parquet')
  This client rewrites that: registers the file as table t0, t1, ...
                             then executes the rewritten SQL.

Drop-in replacement
-------------------
execute_query_sync(sql, connection_string, max_rows, container_name)
  → same return type as duckdb_client.execute_query_sync
  → (rows: list[dict], total: int)
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import re
import time

import pyarrow as pa
from datafusion import SessionContext
from datafusion.object_store import MicrosoftAzure

from app.core.logger import chat_logger
from app.core import metrics

# ── Stdout/stderr suppressor ─────────────────────────────────────────────────
# DataFusion prints ~150 "Registered UDF 'xxx'" lines to stderr on every new
# SessionContext. We silence fd 1+2 during UDF registration only.
# If an exception occurs, the captured stderr is logged so diagnostics survive.
@contextlib.contextmanager
def _silence_datafusion_noise():
    """Silence stdout+stderr at the fd level during SessionContext setup only.

    Captures stderr to a pipe instead of /dev/null so that if an exception
    escapes the with-block we can log the captured output for diagnosis.
    """
    import io
    # Capture stderr; redirect stdout to /dev/null (it only has UDF spam)
    r_fd, w_fd = os.pipe()
    saved_out = os.dup(1)
    saved_err = os.dup(2)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 1)   # stdout  → /dev/null
    os.dup2(w_fd, 2)      # stderr  → our pipe
    os.close(devnull)
    os.close(w_fd)
    exc_info = None
    try:
        yield
    except Exception:
        import sys
        exc_info = sys.exc_info()
        raise
    finally:
        os.dup2(saved_out, 1)
        os.dup2(saved_err, 2)
        os.close(saved_out)
        os.close(saved_err)
        # Drain the pipe (non-blocking)
        os.set_blocking(r_fd, False)
        try:
            captured = os.read(r_fd, 65536)
        except BlockingIOError:
            captured = b""
        os.close(r_fd)
        if exc_info is not None and captured:
            chat_logger.warning(
                "datafusion_stderr",
                captured=captured.decode("utf-8", errors="replace")[:2000],
            )


# ── Object store cache ────────────────────────────────────────────────────────
# MicrosoftAzure objects are pure configuration — no mutable state, safe to
# cache across requests. Avoids re-parsing credentials on every query.
# Key: md5(connection_string + ":" + container_name)
# Capped at 256 entries — evict oldest when full (simple FIFO sufficient here).
_STORE_CACHE_MAX = 256
_store_cache: dict[str, MicrosoftAzure] = {}

# ── SQL patterns ──────────────────────────────────────────────────────────────
# Matches read_parquet('az://...') and read_csv_auto('az://...') with any
# extra keyword args inside the call (e.g., hive_partitioning=true).
_AZ_PATH_PATTERN = re.compile(
    r"(?:read_parquet|read_csv_auto)\s*\(\s*'(az://[^']+)'[^)]*\)",
    re.IGNORECASE,
)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _parse_connection_string(cs: str) -> tuple[str, str]:
    """Extract AccountName and AccountKey from an Azure connection string.

    Connection strings look like:
      DefaultEndpointsProtocol=https;AccountName=foo;AccountKey=bar==;EndpointSuffix=...

    AccountKey values are base64 and may contain trailing '=' padding.
    We split each segment on the FIRST '=' only (maxsplit=1) to preserve them.
    """
    parts: dict[str, str] = {}
    for segment in cs.split(";"):
        if "=" in segment:
            k, v = segment.split("=", 1)
            parts[k.strip()] = v.strip()

    account = parts.get("AccountName", "")
    key = parts.get("AccountKey", "")
    if not account or not key:
        raise ValueError(
            f"Azure connection string missing AccountName or AccountKey. "
            f"Keys found: {list(parts.keys())}"
        )
    return account, key


def _get_object_store(connection_string: str, container_name: str) -> MicrosoftAzure:
    """Return a cached MicrosoftAzure object store for this container.

    Object stores are created once per unique (connection_string, container_name)
    pair and reused across requests. They hold no query state — caching is safe.
    """
    cache_key = hashlib.md5(
        f"{connection_string}:{container_name}".encode()
    ).hexdigest()

    if cache_key not in _store_cache:
        account_name, account_key = _parse_connection_string(connection_string)
        # Evict oldest entry if cache is full
        if len(_store_cache) >= _STORE_CACHE_MAX:
            oldest_key = next(iter(_store_cache))
            del _store_cache[oldest_key]
        _store_cache[cache_key] = MicrosoftAzure(
            container_name,
            account=account_name,
            access_key=account_key,
        )

    return _store_cache[cache_key]


def _extract_container_from_sql(sql: str) -> str | None:
    """Extract the Azure container name from the first az:// URL in the SQL.

    Used as a fallback when container_name is not explicitly provided.
    az://container_name/blob_path → returns 'container_name'
    """
    m = re.search(r"az://([^/'\s]+)/", sql, re.IGNORECASE)
    return m.group(1) if m else None


def _rewrite_sql(sql: str) -> tuple[str, list[str]]:
    """Rewrite read_parquet/read_csv_auto calls to table aliases.

    read_parquet('az://container/file.parquet')  →  t0
    read_csv_auto('az://container/file.csv')     →  t1

    Returns:
        rewritten_sql: SQL with all read_parquet/read_csv_auto replaced by t0, t1, ...
        paths: list of az:// paths in alias order (t0=paths[0], t1=paths[1], ...)
    """
    paths: list[str] = []
    path_to_alias: dict[str, str] = {}

    def _replace(m: re.Match) -> str:
        path = m.group(1)
        if path not in path_to_alias:
            alias = f"t{len(paths)}"
            paths.append(path)
            path_to_alias[path] = alias
        return path_to_alias[path]

    rewritten = _AZ_PATH_PATTERN.sub(_replace, sql)
    return rewritten, paths


def _json_safe(rows: list[dict]) -> list[dict]:
    """Ensure all row values are JSON-serializable."""
    safe = []
    for row in rows:
        safe.append({
            k: v.isoformat() if hasattr(v, "isoformat") else
               (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
            for k, v in row.items()
        })
    return safe


def execute_query_sync(
    sql: str,
    connection_string: str,
    max_rows: int = 1000,
    container_name: str | None = None,
) -> tuple[list[dict], int]:
    """Execute SQL using DataFusion. Drop-in replacement for duckdb_client.execute_query_sync.

    Each call creates a fresh SessionContext — zero shared state between
    concurrent queries. This is the core fix for the DuckDB serialization problem.

    Args:
        sql:               SQL query — same format as DuckDB (read_parquet('az://...'))
        connection_string: Azure storage connection string (plain, already decrypted)
        max_rows:          Cap on returned rows (total count still reported accurately)
        container_name:    Azure container name. If None, extracted from az:// URL in SQL.

    Returns:
        (rows, total_row_count) — rows is a list of dicts, capped at max_rows.
    """
    start = time.perf_counter()
    metrics.inc("query_queue_depth")
    metrics.inc("query_total")
    chat_logger.info(
        "datafusion",
        operation="execute_query",
        status="started",
        sql_preview=sql[:300],
    )

    # ── Resolve container name ────────────────────────────────────────────────
    resolved_container = container_name or _extract_container_from_sql(sql)
    if not resolved_container:
        metrics.dec("query_queue_depth")
        raise ValueError(
            "Cannot resolve Azure container name. Pass container_name explicitly "
            "or ensure SQL contains an az://container_name/... path."
        )

    # ── Rewrite SQL: read_parquet('az://...') → t0, t1, ... ──────────────────
    rewritten_sql, paths = _rewrite_sql(sql)

    try:
        with _silence_datafusion_noise():
            # ── Fresh SessionContext per request — ZERO shared state ──────────
            ctx = SessionContext()

            # ── Register Azure object store once per context ──────────────────
            store = _get_object_store(connection_string, resolved_container)
            ctx.register_object_store(f"az://{resolved_container}/", store)

            # ── Register each file as a named table ───────────────────────────
            for i, path in enumerate(paths):
                alias = f"t{i}"
                if path.lower().endswith(".parquet"):
                    ctx.register_parquet(alias, path)
                else:
                    # CSV fallback — DataFusion infers schema
                    ctx.register_csv(alias, path, has_header=True)

        # Silence scope ends here — query execution stderr is NOT suppressed
        # so genuine DataFusion error messages survive.
        # ── Execute SQL ───────────────────────────────────────────────────
        t_exec = time.perf_counter()
        result_batches = ctx.sql(rewritten_sql).collect()
        exec_ms = _ms(t_exec)

        # ── Convert Arrow RecordBatches → Python dicts ────────────────────────
        t_conv = time.perf_counter()
        if result_batches:
            table = pa.Table.from_batches(result_batches)
        else:
            table = pa.table({})

        total = len(table)
        rows = _json_safe(
            table.slice(0, max_rows).to_pylist()
        )
        conv_ms = _ms(t_conv)

        # ctx goes out of scope here → garbage collected → nothing persists

        total_ms = _ms(start)
        metrics.dec("query_queue_depth")
        metrics.record_query_duration(total_ms)
        chat_logger.info(
            "datafusion",
            operation="execute_query",
            status="done",
            row_count=len(rows),
            total_rows=total,
            truncated=total > max_rows,
            exec_ms=exec_ms,
            convert_ms=conv_ms,
            total_ms=total_ms,
        )
        return rows, total

    except Exception:
        metrics.dec("query_queue_depth")
        metrics.inc("query_errors")
        chat_logger.exception(
            "datafusion",
            operation="execute_query",
            status="error",
            sql_preview=sql[:300],
            total_ms=_ms(start),
        )
        raise
