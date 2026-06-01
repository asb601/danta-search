"""Live read-only org Postgres data source.

PURPOSE:
  Execute a SINGLE read-only SELECT against an organization's own Postgres
  database (the DSN is stored encrypted in OrgAISettings.postgres_url and
  resolved per-request by org_ai_resolver.resolve_org_ai_settings). This lets
  the agent query the org's live operational tables in addition to the
  ingested Parquet catalog.

READ-ONLY ENFORCEMENT (defence in depth — both layers always apply):
  1. SQL validation: reject anything that is not a single read-only SELECT.
     No DDL/DML, no multiple statements, no forbidden keywords (DROP, DELETE,
     INSERT, UPDATE, CREATE, ALTER, TRUNCATE, COPY, ... — same intent as
     execution_guards / sql_safety). A CTE that starts with WITH is allowed
     only when it does not contain a write keyword.
  2. Transaction level: the query runs inside a `READ ONLY` transaction
     (`SET TRANSACTION READ ONLY`) on a connection with a statement_timeout,
     so even if validation were bypassed the engine itself rejects writes.

SAFETY:
  - The DSN is NEVER logged. On connect failure a clear, DSN-free error is
    raised.
  - Results are hard-capped (LIMIT injected if absent + max_rows fetch cap).
  - Introspection is cached in-process with a short TTL keyed by a hash of the
    DSN (never the DSN itself).
"""
from __future__ import annotations

import hashlib
import re
import time

import asyncpg

from app.core.config import get_settings
from app.core.logger import chat_logger

# ── Read-only SQL validation ─────────────────────────────────────────────────

# Same intent as app.agent.tools.sql_safety._FORBIDDEN / execution_guards:
# any of these as a whole word means the statement is not a pure read.
_FORBIDDEN: tuple[str, ...] = (
    "DROP", "DELETE", "UPDATE", "INSERT", "CREATE", "ALTER", "TRUNCATE",
    "COPY", "GRANT", "REVOKE", "MERGE", "CALL", "EXEC", "EXECUTE", "DO",
    "VACUUM", "ANALYZE", "REINDEX", "REFRESH", "SET", "RESET", "LOCK",
    "COMMENT", "SECURITY", "PREPARE", "DEALLOCATE", "LISTEN", "NOTIFY",
    "FETCH", "MOVE", "DECLARE", "INTO",
)
_FORBIDDEN_RE = re.compile(r"\b(" + "|".join(_FORBIDDEN) + r")\b", re.IGNORECASE)
_LIMIT_RE = re.compile(r"\bLIMIT\b", re.IGNORECASE)


class OrgDBError(ValueError):
    """Raised when an org-db query is rejected or the connection fails.

    The message is safe to surface to the LLM as a tool error — it never
    contains the DSN or any credential material.
    """


def _strip_sql(sql: str) -> str:
    """Strip a single trailing semicolon and surrounding whitespace."""
    return (sql or "").strip().rstrip(";").strip()


def validate_readonly_select(sql: str) -> str:
    """Validate `sql` is a single read-only SELECT and return the cleaned SQL.

    Raises OrgDBError on any violation. This is layer 1 of the read-only
    enforcement (the READ ONLY transaction is layer 2).
    """
    cleaned = _strip_sql(sql)
    if not cleaned:
        raise OrgDBError("Empty query. Provide a single read-only SELECT statement.")

    # Reject multiple statements (a semicolon remaining after the strip means
    # there is a second statement).
    if ";" in cleaned:
        raise OrgDBError(
            "Multiple SQL statements are not allowed. Submit a single read-only SELECT."
        )

    lowered_head = cleaned.lstrip("(").lstrip().lower()
    if not (lowered_head.startswith("select") or lowered_head.startswith("with")):
        raise OrgDBError(
            "Only read-only SELECT queries are allowed (a leading WITH is permitted)."
        )

    match = _FORBIDDEN_RE.search(cleaned)
    if match:
        raise OrgDBError(
            f"Forbidden keyword '{match.group(1).upper()}' detected. "
            "Only read-only SELECT queries are allowed."
        )
    return cleaned


def _cap_sql(sql: str, max_rows: int) -> str:
    """Append a LIMIT cap if the query does not already declare one."""
    if _LIMIT_RE.search(sql):
        return sql
    return f"{sql}\nLIMIT {int(max_rows)}"


def _dsn_fingerprint(dsn: str) -> str:
    """Stable, non-reversible cache key for a DSN (never logs the DSN)."""
    return hashlib.sha256((dsn or "").encode("utf-8")).hexdigest()


# ── Execution ────────────────────────────────────────────────────────────────

async def execute_readonly(
    dsn: str,
    sql: str,
    max_rows: int | None = None,
) -> tuple[list[dict], int]:
    """Run a single read-only SELECT and return (rows, row_count).

    Rows are capped at `max_rows` (default ORG_DB_MAX_ROWS). The query runs
    inside a READ ONLY transaction on a connection with a statement timeout.
    """
    settings = get_settings()
    cap = max(1, int(max_rows or settings.ORG_DB_MAX_ROWS))
    timeout = max(1, int(settings.ORG_DB_QUERY_TIMEOUT_SECONDS))

    # Layer 1: SQL validation.
    cleaned = validate_readonly_select(sql)
    capped_sql = _cap_sql(cleaned, cap)

    start = time.perf_counter()
    chat_logger.info("org_db", operation="execute_readonly", status="started",
                     sql_preview=capped_sql[:300])

    try:
        conn = await asyncpg.connect(dsn, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — never leak the DSN
        chat_logger.warning("org_db", operation="execute_readonly",
                            status="connect_failed", error=type(exc).__name__)
        raise OrgDBError(
            "Could not connect to the organization's live database."
        ) from None

    try:
        # Enforce a hard server-side statement timeout for this session.
        await conn.execute(f"SET statement_timeout = {timeout * 1000}")
        # Layer 2: READ ONLY transaction — the engine rejects any write.
        async with conn.transaction(readonly=True):
            records = await conn.fetch(capped_sql, timeout=timeout)
    except OrgDBError:
        raise
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)[:300]
        chat_logger.warning("org_db", operation="execute_readonly",
                            status="query_failed", error=msg)
        raise OrgDBError(f"Query failed against the live database: {msg}") from None
    finally:
        await conn.close()

    rows = [dict(r) for r in records[:cap]]
    duration_ms = round((time.perf_counter() - start) * 1000, 2)
    chat_logger.info("org_db", operation="execute_readonly", status="done",
                     row_count=len(rows), duration_ms=duration_ms)
    return rows, len(rows)


# ── Introspection (schema discovery for the org's live DB) ───────────────────

# Short in-process TTL cache keyed by a hash of the DSN. Value is the
# introspection dict; we also store the insert time for TTL expiry.
_INTROSPECT_TTL_SECONDS = 300
_introspect_cache: dict[str, tuple[float, dict]] = {}

_INTROSPECT_SQL = """
SELECT c.table_schema, c.table_name, c.column_name, c.data_type
FROM information_schema.columns c
JOIN information_schema.tables t
  ON t.table_schema = c.table_schema AND t.table_name = c.table_name
WHERE c.table_schema NOT IN ('pg_catalog', 'information_schema')
  AND c.table_schema NOT LIKE 'pg_%'
  AND t.table_type IN ('BASE TABLE', 'VIEW')
ORDER BY c.table_schema, c.table_name, c.ordinal_position
"""


async def introspect(dsn: str) -> dict[str, list[dict]]:
    """Return {"schema.table": [{"column": ..., "type": ...}, ...]} for the DSN.

    Only non-system schemas are introspected. Results are cached in-process
    with a short TTL keyed by a hash of the DSN. The DSN is never logged.
    """
    key = _dsn_fingerprint(dsn)
    now = time.monotonic()
    cached = _introspect_cache.get(key)
    if cached and (now - cached[0]) < _INTROSPECT_TTL_SECONDS:
        return cached[1]

    timeout = max(1, int(get_settings().ORG_DB_QUERY_TIMEOUT_SECONDS))
    try:
        conn = await asyncpg.connect(dsn, timeout=timeout)
    except Exception as exc:  # noqa: BLE001 — never leak the DSN
        chat_logger.warning("org_db", operation="introspect",
                            status="connect_failed", error=type(exc).__name__)
        raise OrgDBError(
            "Could not connect to the organization's live database."
        ) from None

    try:
        records = await conn.fetch(_INTROSPECT_SQL, timeout=timeout)
    except Exception as exc:  # noqa: BLE001
        chat_logger.warning("org_db", operation="introspect",
                            status="query_failed", error=str(exc)[:200])
        raise OrgDBError("Could not introspect the live database schema.") from None
    finally:
        await conn.close()

    schema: dict[str, list[dict]] = {}
    for r in records:
        table_key = f"{r['table_schema']}.{r['table_name']}"
        schema.setdefault(table_key, []).append(
            {"column": r["column_name"], "type": r["data_type"]}
        )

    _introspect_cache[key] = (now, schema)
    chat_logger.info("org_db", operation="introspect", status="done",
                     table_count=len(schema))
    return schema
