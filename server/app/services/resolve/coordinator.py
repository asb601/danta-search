"""Brain-resolve coordinator — the flag-gated, additive query seam.

This is the orchestration entry point for the decided query architecture: a
narrow, twin-aware retrieved slice handed to the LLM BRAIN, which emits a typed
contract that is value-verified, with arithmetic done deterministically by
SQL/DataFusion. The brain never computes a number, never selects from the whole
corpus, and abstains on genuine ties rather than guessing.

Flow:
  1. ``decompose_question`` — one gpt-4o-mini call → typed intent (entities + …).
  2. ``search_per_entity`` — per-entity, container-scoped, twin-together retrieval
     of a small candidate slice (the right small slice, not the whole catalog).
  3. SINGLE entity → ``brain.brain_resolve`` over that slice → a verified
     single-table contract + deterministic SQL.
     MULTI entity → resolve each entity's table, then VERIFY a join between them
     from value evidence (``verification.verify_join``); render the joined
     aggregate ONLY if a join verifies — otherwise ABSTAIN.
  4. Execute through the SAME logical→physical canonicalizer + engine router the
     agent's run_sql and the existing RESOLVE seam use.

Returns a ``run_agent_query``-shaped payload (route="brain_resolve") when it
resolves and executes, or ``None`` to fall through to the existing path. NEVER
raises. Gated at the calling site by ``BRAIN_RESOLVE_ENABLED`` (default False).
"""
from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.models.column_key_registry import ColumnKeyRegistry

logger = structlog.get_logger("resolve.coordinator")

# How many candidate join columns to test per side before giving up. Bounds the
# verify_join fan-out; it is a safety cap, not a business threshold.
_MAX_JOIN_COLS_PER_SIDE = 12


def _quote(name: str) -> str:
    """ANSI double-quote an identifier (DataFusion is case-sensitive; the parquet
    schema is uppercase). Embedded quotes doubled. Pure — same quoting the brain
    emitter uses so the joined SQL renders identically."""
    return '"' + str(name).replace('"', '""') + '"'


async def _candidate_key_columns(
    db: AsyncSession, container_id: str, blob_path: str,
) -> list[str]:
    """Candidate join-key columns for a blob, from the precomputed registry.

    Reads ColumnKeyRegistry only (precomputed value-evidence) — no schema probe.
    Returns column names ordered by uniqueness (PK-likely first) so verify_join
    tests the most promising pairs first. Never raises (rollback + [] on error)."""
    try:
        rows = (
            await db.execute(
                select(ColumnKeyRegistry.column_name, ColumnKeyRegistry.unique_rate)
                .where(ColumnKeyRegistry.container_id == container_id)
                .where(ColumnKeyRegistry.blob_path == blob_path)
                .order_by(ColumnKeyRegistry.unique_rate.desc())
            )
        ).all()
    except Exception as exc:  # noqa: BLE001 — never raise; roll back and degrade
        await db.rollback()
        logger.warning("key_columns_query_error", error=str(exc)[:200])
        return []
    return [str(name) for name, _ur in rows[:_MAX_JOIN_COLS_PER_SIDE]]


def _blob_for_table(file_id: str, file_identity_map) -> str | None:
    """Resolve the representative blob_path for a resolved table's file_id from
    the request identity map (the same map the executor trusts)."""
    if file_identity_map is None:
        return None
    identity = getattr(file_identity_map, "by_id", {}).get(file_id)
    return getattr(identity, "blob_path", None) if identity is not None else None


async def _resolve_single(db, question, candidates, time_window):
    """SINGLE-entity path: hand the slice to the brain. Returns the brain's
    ``{sql, table, grain, measure, reason}`` or None (abstain)."""
    from app.services.resolve.brain import brain_resolve  # noqa: PLC0415
    return await brain_resolve(db, question, candidates, time_window)


async def _resolve_join(
    db: AsyncSession, container_id: str, question, entity_slices, time_window,
    file_identity_map,
):
    """MULTI-entity path: resolve each entity's single best table via the brain,
    then VERIFY a join between the two from value evidence. Renders a joined,
    relationship-validated aggregate ONLY when a join verifies; abstains (None)
    otherwise. No arbitrary LLM joins — verify_join is the only authority."""
    from app.services.resolve.brain import brain_resolve  # noqa: PLC0415
    from app.services.resolve.verification import verify_join  # noqa: PLC0415

    # Resolve one table per entity (the brain picks canonical within each slice).
    resolved: list[dict] = []
    for entity, candidates in entity_slices.items():
        if not candidates:
            continue
        out = await brain_resolve(db, question, candidates, time_window)
        if not out:
            continue
        # Map the chosen logical table back to its file_id/blob via the slice.
        chosen = next(
            (c for c in candidates if str(c.get("table", "")).upper() == str(out["table"]).upper()),
            None,
        )
        if not chosen:
            continue
        blob = _blob_for_table(chosen["file_id"], file_identity_map)
        if not blob:
            continue
        resolved.append({"table": out["table"], "file_id": chosen["file_id"], "blob": blob})

    if len(resolved) < 2:
        logger.info("join_abstain_insufficient_tables", resolved=len(resolved))
        return None

    # Only the first two distinct tables are joined (a narrow, two-table seam).
    a, b = resolved[0], resolved[1]
    if a["blob"] == b["blob"]:
        logger.info("join_abstain_same_table")
        return None

    cols_a = await _candidate_key_columns(db, container_id, a["blob"])
    cols_b = await _candidate_key_columns(db, container_id, b["blob"])
    if not cols_a or not cols_b:
        logger.info("join_abstain_no_key_columns")
        return None

    # Test candidate key pairs; accept the first VERIFIED 1:N edge. This is the
    # only place a join is allowed to come from — value evidence, never the LLM.
    for ca in cols_a:
        for cb in cols_b:
            verdict = await verify_join(db, container_id, a["blob"], ca, b["blob"], cb)
            if verdict.verified:
                # PK side drives the GROUP BY grain; render a deterministic,
                # fully-quoted joined row-count over the relationship-validated key.
                if verdict.pk_side == a["blob"]:
                    pk_table, pk_col, fk_table, fk_col = a["table"], ca, b["table"], cb
                else:
                    pk_table, pk_col, fk_table, fk_col = b["table"], cb, a["table"], ca
                sql = (
                    f'SELECT {_quote(pk_table)}.{_quote(pk_col)} AS {_quote(pk_col.lower())}, '
                    f'COUNT(*) AS "match_count"\n'
                    f'FROM {_quote(pk_table)}\n'
                    f'JOIN {_quote(fk_table)} '
                    f'ON {_quote(pk_table)}.{_quote(pk_col)} = {_quote(fk_table)}.{_quote(fk_col)}\n'
                    f'GROUP BY {_quote(pk_table)}.{_quote(pk_col)}\n'
                    f'ORDER BY "match_count" DESC'
                )
                logger.info(
                    "join_verified",
                    pk_table=pk_table, pk_col=pk_col, fk_table=fk_table, fk_col=fk_col,
                    containment=round(verdict.containment, 4),
                )
                return {
                    "sql": sql,
                    "table": pk_table,
                    "grain": "entity",
                    "measure": "COUNT(*)",
                    "reason": f"verified join {pk_table}.{pk_col} <- {fk_table}.{fk_col}",
                    "files_used": [pk_table, fk_table],
                }

    logger.info("join_abstain_no_verified_edge")
    return None


async def brain_answer(
    question: str,
    db: AsyncSession,
    container_id: str,
    ctx: dict,
    initial_state: dict,
    req_id: str,
) -> dict | None:
    """Brain-resolve a question end-to-end, or ``None`` to fall through.

    Decompose → per-entity twin-aware retrieval → brain (single) or verified-join
    (multi) → canonicalize + execute through the same engine router the agent
    uses. Returns a run_agent_query-shaped payload (route="brain_resolve") or
    None. NEVER raises.
    """
    try:
        from app.services.resolve.decompose import decompose_question  # noqa: PLC0415
        from app.services.resolve.search import search_per_entity  # noqa: PLC0415

        effective_container_id = ctx.get("resolved_container_id") or container_id
        if not effective_container_id:
            logger.info("brain_answer_no_container")
            return None

        decomposed = await decompose_question(question, ctx.get("intent_plan"))
        if not decomposed:
            return None
        entities = decomposed.get("entities") or []
        if not entities:
            return None
        # The grain entity is the GROUP-BY axis (e.g. "...by customer"), NOT a
        # separate table to join. Drop it from the table-SUBJECT entities so
        # "X by customer" resolves as a SINGLE table grained by customer, instead
        # of a spurious customer⋈X join (which would abstain for lack of a verified
        # edge). Falls back to the full list if dropping leaves nothing.
        grain_entity = decomposed.get("grain_entity")
        subject_entities = [e for e in entities if e and e != grain_entity] or entities

        file_identity_map = ctx.get("file_identity_map")
        entity_slices = await search_per_entity(
            db,
            effective_container_id,
            subject_entities,
            top_k=9,
            min_score=0.0,
            file_identity_map=file_identity_map,
        )
        # The brain resolves time grain from the evidence itself; the deterministic
        # relative-window math is not invented here, so the brain runs without an
        # externally pre-resolved window.
        time_window = None

        if len(subject_entities) <= 1:
            slice_for_entity = entity_slices.get(subject_entities[0], []) if subject_entities else []
            if not slice_for_entity:
                logger.info("brain_answer_empty_slice", entity=subject_entities[0] if subject_entities else "")
                return None
            resolved = await _resolve_single(db, question, slice_for_entity, time_window)
        else:
            resolved = await _resolve_join(
                db, effective_container_id, question, entity_slices, time_window,
                file_identity_map,
            )

        if not resolved or not resolved.get("sql"):
            return None

        # ── Execute through the SAME path the existing RESOLVE seam uses ──────────
        # canonicalize logical→physical, then the engine router; identical
        # store-cleanup so a brain answer leaves no request store behind.
        from app.agent.tools.sql import _execute as _brain_execute  # noqa: PLC0415
        from app.services.logical_sql import canonicalize_logical_sql  # noqa: PLC0415

        canon = canonicalize_logical_sql(
            resolved["sql"],
            ctx["file_identity_map"],
            allowed_file_ids=ctx["allowed_file_ids"],
        )
        rows, total = await asyncio.to_thread(
            _brain_execute,
            canon.executable_sql,
            initial_state["connection_string"],
            ctx["container_name"],
            20,
        )
        rows = rows or []
        total = total if total is not None else len(rows)

        # Store cleanup — same as _resolve_contract_payload's success path.
        try:
            from app.agent.graph.graph import _request_stores, _stores_lock  # noqa: PLC0415
            with _stores_lock:
                _request_stores.pop(req_id, None)
        except Exception:  # noqa: BLE001 — cleanup must never break the answer
            pass

        # Business-readable prose DERIVED from the resolved contract (the measure
        # expression and table), never hardcoded. The number itself comes from SQL.
        measure = str(resolved.get("measure") or "result")
        table = str(resolved.get("table") or "")
        if rows:
            _shown = f" (showing the top {len(rows)})" if total > len(rows) else ""
            answer = f"{total:,} rows for {measure} over {table}{_shown}."
        else:
            answer = f"No rows for {measure} over {table}."

        files_used = resolved.get("files_used") or ([table] if table else [])

        trace = ctx.get("trace")
        if trace is not None:
            try:
                trace.set_execution_outcome(rows=len(rows), total=total, duration_ms=0.0)
                trace.emit()
            except Exception:  # noqa: BLE001 — trace must never break the answer
                pass

        logger.info(
            "brain_answer",
            table=table,
            measure=measure,
            grain=resolved.get("grain"),
            row_count=len(rows),
            total_rows=total,
            reason=str(resolved.get("reason", ""))[:160],
        )
        return {
            "answer": answer,
            "data": rows,
            "chart": None,
            "route": "brain_resolve",
            "row_count": total,
            "files_used": files_used,
            "tool_calls": 0,
            "retrieved_files": ctx.get("catalog_len", 0),
            "total_files": ctx.get("total_files", 0),
        }
    except Exception as exc:  # noqa: BLE001 — never raise; caller falls through
        logger.warning("brain_answer_seam_error", error=str(exc)[:200])
        return None
