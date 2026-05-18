import asyncio
import hashlib
import os
import threading
import time

_CA_BUNDLE = "/etc/ssl/certs/ca-certificates.crt"
if os.path.exists(_CA_BUNDLE):
    os.environ["CURL_CA_BUNDLE"] = _CA_BUNDLE
    os.environ["SSL_CERT_FILE"] = _CA_BUNDLE
    os.environ["REQUESTS_CA_BUNDLE"] = _CA_BUNDLE

import duckdb

from app.core.config import get_settings
from app.core.logger import ingest_logger, chat_logger
from app.services.ingestion_config import null_tokens


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


def _duckdb_string(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _duckdb_string_list(values: tuple[str, ...]) -> str:
    return "[" + ", ".join(_duckdb_string(value) for value in values) + "]"


_thread_local = threading.local()


def _get_connection(connection_string: str) -> duckdb.DuckDBPyConnection:
    key = hashlib.md5(connection_string.encode()).hexdigest()
    cache: dict = getattr(_thread_local, "connections", None)
    if cache is None:
        _thread_local.connections = {}
        cache = _thread_local.connections
    if key not in cache:
        conn = duckdb.connect()
        # INSTALL azure IF NOT EXISTS requires DuckDB ≥ 0.10; fall back to bare
        # INSTALL (re-installs if not cached, but never raises a syntax error)
        # and then suppress the "already installed" error for bare INSTALL.
        try:
            conn.execute("INSTALL azure IF NOT EXISTS;")
        except Exception:
            try:
                conn.execute("INSTALL azure;")
            except Exception:
                pass  # already installed — safe to ignore
        conn.execute("LOAD azure;")
        conn.execute("SET azure_transport_option_type = 'curl';")
        safe_conn = connection_string.replace("'", "''")
        conn.execute(f"SET azure_storage_connection_string='{safe_conn}';")
        cache[key] = conn
    return cache[key]


def _clear_connection(connection_string: str) -> None:
    key = hashlib.md5(connection_string.encode()).hexdigest()
    cache: dict = getattr(_thread_local, "connections", {})
    cache.pop(key, None)


async def sample_file(
    blob_path: str, connection_string: str, container_name: str
) -> dict:
    def _run() -> dict:
        try:
            settings = get_settings()
            sample_rows = max(1, int(settings.INGEST_DUCKDB_SAMPLE_ROWS))
            unique_values_limit = max(0, int(settings.INGEST_COLUMN_UNIQUE_VALUES_LIMIT))
            sample_values_limit = max(0, int(settings.INGEST_COLUMN_SAMPLE_VALUES_LIMIT))
            nullstr = _duckdb_string_list(null_tokens())
            conn = _get_connection(connection_string)
            azure_path = f"az://{container_name}/{blob_path}".replace("'", "''")

            t = time.perf_counter()
            df = conn.execute(
                f"""
                SELECT * FROM read_csv_auto(
                    '{azure_path}',
                    sample_size={sample_rows},
                    null_padding=true,
                    ignore_errors=true,
                    nullstr={nullstr}
                ) LIMIT {sample_rows}
                """
            ).df()
            read_ms = _ms(t)

            columns_info: list[dict] = []
            for col in df.columns:
                unique_vals = df[col].dropna().unique().tolist()[:unique_values_limit]
                sample_vals = df[col].dropna().head(sample_values_limit).tolist()
                columns_info.append(
                    {
                        "name": col,
                        "type": str(df[col].dtype),
                        "sample_values": [str(v) for v in sample_vals],
                        "unique_values": [str(v) for v in unique_vals],
                    }
                )

            return {
                "columns_info": columns_info,
                "sample_rows": _json_safe(df.astype(object).fillna("").to_dict("records")),
                "row_count": len(df),
                "row_count_approx": len(df) == sample_rows,
                "column_names": list(df.columns),
                "_sample_rows": sample_rows,
                "_read_ms": read_ms,
            }
        except Exception:
            _clear_connection(connection_string)
            raise

    start = time.perf_counter()
    ingest_logger.info("duckdb", operation="sample_file", status="started",
                       blob_path=blob_path)
    result = await asyncio.to_thread(_run)
    approx = result.pop("row_count_approx")
    sample_rows = result.pop("_sample_rows")
    read_ms = result.pop("_read_ms")
    ingest_logger.info("duckdb", operation="sample_file", status="done",
                       blob_path=blob_path,
                       columns=len(result["columns_info"]),
                       row_count=result["row_count"],
                       row_count_note=f"{sample_rows}+ (sample limit)" if approx else "exact",
                       duration_ms=read_ms)
    return result


def _json_safe(rows: list[dict]) -> list[dict]:
    safe = []
    for row in rows:
        safe.append({
            k: v.isoformat() if hasattr(v, "isoformat") else
               (str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v)
            for k, v in row.items()
        })
    return safe


def execute_query_sync(
    sql: str, connection_string: str, max_rows: int | None = None,
) -> tuple[list[dict], int]:
    """Synchronous SQL execution. Returns (rows, total_row_count). Rows capped at max_rows."""
    resolved_max_rows = max(1, int(max_rows or get_settings().INGEST_DUCKDB_QUERY_MAX_ROWS))
    start = time.perf_counter()
    chat_logger.info("duckdb", operation="execute_query", status="started",
                     sql_preview=sql[:300])
    try:
        t_conn = time.perf_counter()
        conn = _get_connection(connection_string)
        conn_ms = _ms(t_conn)

        t_exec = time.perf_counter()
        result = conn.execute(sql).df()
        exec_ms = _ms(t_exec)

        t_conv = time.perf_counter()
        total = len(result)
        rows = _json_safe(result.head(resolved_max_rows).fillna("").to_dict("records"))
        conv_ms = _ms(t_conv)

        chat_logger.info("duckdb", operation="execute_query", status="done",
                         row_count=len(rows), total_rows=total,
                         truncated=total > resolved_max_rows,
                         conn_ms=conn_ms, exec_ms=exec_ms,
                         convert_ms=conv_ms, total_ms=_ms(start))
        return rows, total
    except Exception:
        _clear_connection(connection_string)
        raise


async def execute_query(
    sql: str, connection_string: str, timeout_seconds: int | None = None,
    max_rows: int | None = None,
) -> tuple[list[dict], int]:
    """Async SQL execution. Returns (rows, total_row_count). Rows capped at max_rows."""
    resolved_timeout = max(1, int(timeout_seconds or get_settings().INGEST_DUCKDB_QUERY_TIMEOUT_SECONDS))
    return await asyncio.wait_for(
        asyncio.to_thread(execute_query_sync, sql, connection_string, max_rows),
        timeout=resolved_timeout,
    )


def _resolve_data_path(
    blob_path: str, connection_string: str, container_name: str,
    parquet_blob_path: str | None,
) -> str:
    if parquet_blob_path:
        return f"az://{container_name}/{parquet_blob_path}"
    return f"az://{container_name}/{blob_path}"
