"""v2 RESOLVE binder — DB-backed, deterministic question → BOUND Contract.

This is the BINDING half of the deterministic query brain — the step the
resolver/emitter modules deliberately leave out. ``resolve_metric_query`` is a
pure assembler that takes *already-resolved* arguments (metric, canonical
source, grain key, threshold, display columns); SOMETHING has to resolve those
arguments from the governed metadata FIRST. That something is this module.

``bind_contract_from_db`` reads the governed metric definitions and the
canonical-master election DIRECTLY FROM THE DATABASE (the ``semantic_entities``
``metrics`` JSONB + ``is_canonical_master`` flag) and turns a natural-language
question into a fully-BOUND ``Contract`` via ``resolve_metric_query`` — with NO
hand-fed arguments and NO LLM on the exact-match path.

Design properties (enforced, not aspirational):
  * Exact-match, deterministic. The question is matched to a governed metric by
    case-insensitive substring against the metric's OWN synonyms (and its name
    with underscores rendered as spaces). No LLM, no embedding, no fuzzy score.
  * Governed-only. A metric binds only if it is a *governed* entry — it must
    carry ``name`` AND ``synonyms`` AND ``measure`` AND ``grain``. The raw
    measure-only entries (``name``/``column``/``default_aggregation`` only) that
    also live in the same ``metrics`` list are NOT governed and are skipped.
  * Canonical-master verified. The governed metric is read off a specific
    ``semantic_entities`` row; that carrying row must itself be the elected
    ``is_canonical_master`` for the bind to proceed. The election is value-
    evidence-driven and computed at ingestion (``verify_canonical`` →
    ``apply_master_election``); this module only TRUSTS the persisted flag, it
    never re-elects.
  * No hardcoded business terms. Every business identifier — the synonyms, the
    measure column, the default aggregation, the filter predicate, the grain
    key, the source table — comes from the DB metric. The ONLY literals in this
    file are (a) regex patterns for parsing a numeric threshold out of the
    question and (b) the ``_ID`` → ``_NAME`` display-column naming convention,
    both commented at their site.
  * Additive. Nothing here is wired into ``graph.py``; activation is gated by the
    ``RESOLVE_CONTRACT_ENABLED`` flag at the calling site, default False.
"""
from __future__ import annotations

import re
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.semantic_layer import SemanticEntity
from app.services.resolve.contract import Contract
from app.services.resolve.resolver import resolve_metric_query

logger = structlog.get_logger("resolve.binder")

# A governed metric MUST carry these keys. The raw measure-only entries in the
# same `metrics` JSONB list (which only have name/column/default_aggregation) are
# deliberately excluded — they are not governed business definitions.
_GOVERNED_METRIC_KEYS: tuple[str, ...] = ("name", "synonyms", "measure", "grain")


def _sql_value(value: object) -> str:
    """Render a filter value as a safe SQL literal (numeric as-is, else quoted)."""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _build_filter_preds(filt: object) -> tuple[str, ...]:
    """Build QUOTED SQL filter predicates from a governed-metric filter.

    Accepts a structured filter — a dict ``{"column", "op", "value"}`` or a list
    of them — and quotes each column so it matches the uppercase parquet schema.
    A legacy raw-string predicate is passed through unchanged (the author is then
    responsible for quoting). ``None``/empty → no predicates.
    """
    if not filt:
        return ()
    items = filt if isinstance(filt, list) else [filt]
    preds: list[str] = []
    for f in items:
        if isinstance(f, dict) and f.get("column"):
            col = str(f["column"])
            op = str(f.get("op", "=")).strip() or "="
            preds.append(f'"{col}" {op} {_sql_value(f.get("value"))}')
        elif isinstance(f, str) and f.strip():
            preds.append(f.strip())
    return tuple(preds)

# Threshold parsing patterns. These are NUMBER-SHAPE patterns only — they carry
# NO business meaning (no column names, no domain terms). They recognise the
# common ways a question expresses a numeric floor: "over $500,000",
# "more than 500000", "> 500k", "above 500000". The captured group is the raw
# number token (with optional commas / $ / trailing k|m), normalised by
# `_parse_threshold_number`.
_THRESHOLD_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:over|above|more than|greater than|exceed(?:ing|s)?|>=?)\s*\$?\s*"
               r"([0-9][0-9,]*\.?[0-9]*\s*[kKmM]?)"),
    re.compile(r"\$\s*([0-9][0-9,]*\.?[0-9]*\s*[kKmM]?)\s*(?:or more|\+)"),
)

# Multipliers for the optional magnitude suffix on a parsed threshold number.
# Pure number formatting (k = thousand, m = million); no business semantics.
_SUFFIX_MULTIPLIER: dict[str, int] = {"k": 1_000, "m": 1_000_000}


def _parse_threshold_number(token: str) -> int | None:
    """Normalise a captured number token (e.g. '500,000', '500k') to an int.

    Strips ``$`` and commas, applies a trailing k/m multiplier if present, and
    returns the integer value. Pure string→number transform; returns None if the
    token does not resolve to a finite number.
    """
    raw = token.strip().lower().replace("$", "").replace(",", "").replace(" ", "")
    if not raw:
        return None
    multiplier = 1
    if raw[-1] in _SUFFIX_MULTIPLIER:
        multiplier = _SUFFIX_MULTIPLIER[raw[-1]]
        raw = raw[:-1]
    try:
        value = float(raw) * multiplier
    except ValueError:
        return None
    return int(value)


def _extract_threshold(question_lower: str) -> dict[str, Any] | None:
    """Deterministically extract a per-group HAVING threshold from the question.

    Returns ``{"op": ">", "value": <int>}`` if a numeric floor is expressed,
    else ``None``. The operator is always ``>`` — these patterns only recognise
    "over / more than / above" style FLOORS, which is a strictly-greater cut.
    """
    for pattern in _THRESHOLD_PATTERNS:
        match = pattern.search(question_lower)
        if not match:
            continue
        value = _parse_threshold_number(match.group(1))
        if value is not None:
            return {"op": ">", "value": value}
    return None


def _is_governed(metric: Any) -> bool:
    """A metric is governed iff it is a dict carrying every required key and the
    synonyms list is non-empty. Raw measure-only entries fail this check."""
    if not isinstance(metric, dict):
        return False
    if not all(metric.get(k) for k in _GOVERNED_METRIC_KEYS):
        return False
    return bool(metric.get("synonyms"))


def _metric_match_terms(metric: dict[str, Any]) -> list[str]:
    """The case-insensitive match surface for a governed metric: its declared
    synonyms plus its own name with underscores rendered as spaces. Every term
    comes from the metric itself — no external vocabulary."""
    terms: list[str] = []
    for syn in (metric.get("synonyms") or []):
        if isinstance(syn, str) and syn.strip():
            terms.append(syn.strip().lower())
    name = metric.get("name")
    if isinstance(name, str) and name.strip():
        # name with underscores → spaces (e.g. "open_receivables" → "open receivables")
        terms.append(name.strip().lower().replace("_", " "))
    return terms


def _display_cols_for_grain(grain_col: str) -> tuple[str, ...]:
    """Derive an optional display column from the grain key by the ``_ID`` →
    ``_NAME`` convention: a grain key like ``CUSTOMER_ID`` implies a
    human-readable ``CUSTOMER_NAME``. This is a naming CONVENTION, not a
    business term — it is the only structural literal in this module. If the
    grain key does not end in ``_ID`` we add no display column (the emitter
    simply omits it)."""
    if grain_col.upper().endswith("_ID"):
        return (grain_col[: -len("_ID")] + "_NAME",)
    return ()


async def _resolve_container_id(db: AsyncSession, raw: str | None) -> str | None:
    """Resolve a container identifier to the canonical ``ContainerConfig.id``.

    The chat scope passes whatever the caller has — which may be the container's
    display ``name`` or its blob ``container_name`` (e.g. ``test-container01``),
    while ``semantic_entities.container_id`` is the ``ContainerConfig.id`` UUID.
    Match on any of the three and return the canonical id; if nothing matches,
    return the input unchanged so behaviour degrades to the old direct lookup.
    """
    if not raw:
        return raw
    from app.models.container import ContainerConfig  # noqa: PLC0415
    resolved = (
        await db.execute(
            select(ContainerConfig.id)
            .where(
                (ContainerConfig.id == raw)
                | (ContainerConfig.name == raw)
                | (ContainerConfig.container_name == raw)
            )
            .limit(1)
        )
    ).scalar_one_or_none()
    return resolved or raw


async def bind_contract_from_db(
    db: AsyncSession,
    container_id: str,
    question: str,
) -> tuple[Contract | None, str]:
    """Bind a question to a fully-BOUND ``Contract`` from governed DB metadata.

    Pure deterministic exact-match path — NO LLM. Reads governed metrics and the
    canonical-master election from ``semantic_entities`` and assembles the
    contract via ``resolve_metric_query``.

    Returns ``(contract, "bound")`` on success, or ``(None, reason)`` when no
    governed metric matches (``"no_governed_metric_match"``) or the matched
    metric's carrying entity is not an elected canonical master
    (``"no_verified_canonical_master"``).
    """
    q_lower = (question or "").lower()

    # Step 0 — resolve the container identifier (UUID | display name | blob
    # container_name) to the canonical ContainerConfig.id that
    # semantic_entities.container_id uses. The chat scope may pass the NAME.
    container_id = await _resolve_container_id(db, container_id)

    # Step 1 — load every entity in the container that carries metrics. We keep
    # the carrying entity's canonical-master flag with each governed metric, so
    # the master election (Step 3) is data-driven off the persisted flag.
    result = await db.execute(
        select(
            SemanticEntity.entity_name,
            SemanticEntity.metrics,
            SemanticEntity.is_canonical_master,
            SemanticEntity.file_id,
        ).where(
            SemanticEntity.container_id == container_id,
            SemanticEntity.metrics.isnot(None),
        )
    )

    # Flatten into governed-metric records, each tagged with its carrying entity.
    governed: list[dict[str, Any]] = []
    for entity_name, metrics, is_master, file_id in result.all():
        for metric in (metrics or []):
            if not _is_governed(metric):
                # Skip raw measure-only entries that lack synonyms/grain — those
                # are not governed business definitions.
                continue
            governed.append(
                {
                    "metric": metric,
                    "entity_name": entity_name,
                    "is_canonical_master": bool(is_master),
                    "file_id": file_id,
                }
            )

    if not governed:
        return None, "no_governed_metric_match"

    # Step 2 — exact-match the question to a governed metric: case-insensitive,
    # any synonym (or the name with underscores→spaces) is a substring of the
    # lowercased question.
    matched: dict[str, Any] | None = None
    for record in governed:
        terms = _metric_match_terms(record["metric"])
        if any(term in q_lower for term in terms):
            matched = record
            break

    if matched is None:
        return None, "no_governed_metric_match"

    metric = matched["metric"]

    # Step 3 — resolve the canonical source. The governed metric carries its
    # `source`; the carrying entity row must itself be the elected canonical
    # master (value-evidence-driven election persisted at ingestion). A metric
    # with no source, or one carried on a non-master entity, is not verified.
    source = metric.get("source")
    if not source or not matched["is_canonical_master"]:
        logger.info(
            "binder_no_verified_canonical_master",
            metric_name=metric.get("name"),
            source=source,
            is_canonical_master=matched["is_canonical_master"],
        )
        return None, "no_verified_canonical_master"

    # Step 4 — build the resolved-metric dict the resolver expects. Column
    # identifiers are QUOTED so they match the case-sensitive, uppercase parquet
    # schema; the executor lowercases bare identifiers and they would not resolve.
    default_agg = metric.get("default_aggregation") or "SUM"
    measure = metric.get("measure")
    measure_expr = f'{default_agg}("{measure}")'
    filter_preds = _build_filter_preds(metric.get("filter"))
    resolved_metric = {
        "name": metric.get("name"),
        "measure_expr": measure_expr,
        "filter_preds": filter_preds,
        "source": source,
    }

    # Step 5 — grain primary key: the metric's declared grain column.
    grain_col = str(metric.get("grain"))
    grain_pk = (grain_col,)

    # Step 6 — parse an optional per-group HAVING threshold from the question.
    having = _extract_threshold(q_lower)

    # Step 7 — derive an optional display column from the grain key (the
    # `_ID` → `_NAME` convention). The emitter wraps it in MAX() so it rides
    # along without splitting the grain.
    display_cols = _display_cols_for_grain(grain_col)

    # Step 8 — assemble the fully-BOUND contract from the resolved inputs.
    contract = resolve_metric_query(
        question=question,
        metric=resolved_metric,
        canonical_source=str(source),
        grain_pk=grain_pk,
        having=having,
        display_cols=display_cols,
    )

    logger.info(
        "binder_bound_contract",
        metric_name=metric.get("name"),
        source=source,
        grain=grain_col,
        has_threshold=having is not None,
        display_cols=list(display_cols),
    )
    return contract, "bound"
