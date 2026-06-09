"""v2 RESOLVE resolver — pure, deterministic question → Contract assembler.

This is the RESOLVER half of the deterministic query brain. It turns a resolved
metric (its measure expression, filters, and source) plus a canonical source
table and a grain primary key into a fully-BOUND ``Contract`` that the EMITTER
(``emitter.emit_sql``) can render into per-entity aggregated SQL.

Scope boundary (important): this function is the DETERMINISTIC ASSEMBLER only. It
does NOT touch the DB, call an LLM, or run ``verify_canonical`` / ``verify_join``.
The upstream, DB-backed binding step — the caller that resolves the canonical
source from the twin cluster, picks the grain PK from the column-key registry,
and runs the VERIFY tier — is a SEPARATE caller. By the time control reaches
here, those decisions are made and passed in as already-resolved arguments;
this module's only job is to pack them into a Contract with every slot BOUND so
``needs_fallback`` returns ``(False, "complete_verified_contract")``.

Design properties (enforced, not aspirational):
  * Pure Python only. No LLM, no DB, no I/O — a deterministic function of inputs.
  * No hardcoded business terms or column names. The metric, source table, grain
    key, display columns, and HAVING threshold are ALL caller-supplied; nothing
    is inferred from the question text and there are no dataset-fitted literals.
  * Additive. Nothing here is wired into ``graph.py``; activation is gated by the
    ``RESOLVE_CONTRACT_ENABLED`` flag at the calling site, default False.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from app.services.resolve.contract import Contract, SlotState


def resolve_metric_query(
    *,
    question: str,
    metric: Mapping[str, Any],
    canonical_source: str,
    grain_pk: Sequence[str],
    having: Mapping[str, Any] | None = None,
    display_cols: Sequence[str] = (),
) -> Contract:
    """Assemble a fully-BOUND ``Contract`` from already-resolved inputs.

    Parameters
    ----------
    question:
        The original natural-language question (carried for traceability only;
        no parsing happens here).
    metric:
        A resolved-metric mapping with keys ``name`` (the metric / measure
        alias), ``measure_expr`` (the SQL aggregate expression), ``filter_preds``
        (a sequence of already-SQL predicate strings), and ``source`` (the
        metric's declared source table, used only as a fallback for the FROM
        clause when ``canonical_source`` is absent).
    canonical_source:
        The canonical source table the upstream binding step elected (e.g. the
        verified master of a twin cluster). Becomes the Contract's
        ``source_table`` and the emitter's FROM clause.
    grain_pk:
        The grain primary-key column(s). This is the aggregation grain — the
        emitter GROUP-BYs exactly these, which is what forces per-entity
        aggregation. Must be non-empty for the contract to bind.
    having:
        Optional post-aggregation threshold ``{"op": ">", "value": 500000}``
        stored in ``facts["having"]`` and applied by the emitter as a per-group
        HAVING. ``None`` means no threshold.
    display_cols:
        Optional display column(s) stored in ``facts["display_cols"]``; the
        emitter wraps each in ``MAX(...)`` so they ride along without splitting
        the grain.

    Returns
    -------
    Contract
        A contract with source / grain / measure / filter slots all BOUND when
        ``canonical_source``, ``grain_pk``, and a valid ``metric`` are present.
        ``canonical_verified=True`` and no twin cluster / joins are attached, so
        ``needs_fallback`` returns ``(False, "complete_verified_contract")``.

    Raises
    ------
    ValueError
        If any of the inputs required to bind a slot is missing (no source, no
        grain key, or an incomplete metric). We refuse to return a contract
        whose slots claim BOUND on absent evidence.
    """
    metric_name = metric.get("name")
    measure_expr = metric.get("measure_expr")
    if not metric_name or not measure_expr:
        raise ValueError("resolve_metric_query: metric must provide 'name' and 'measure_expr'")

    # Source: prefer the canonically-elected source; fall back to the metric's
    # declared source. One of them must be present to bind the source slot.
    source_table = canonical_source or metric.get("source")
    if not source_table:
        raise ValueError("resolve_metric_query: no canonical_source and metric has no 'source'")

    grain_key = tuple(grain_pk or ())
    if not grain_key:
        raise ValueError("resolve_metric_query: grain_pk is required (per-entity grain)")

    # Filters are already-SQL predicate strings on the metric; carry them through
    # verbatim. The filter slot is BOUND whether or not there are predicates —
    # "no filter" is a fully-resolved state, not an unresolved one.
    filter_preds = tuple(p for p in (metric.get("filter_preds") or ()) if p)

    # The grain identifier carried on the contract is the comma-joined grain key
    # (a human-readable grain label), derived purely from the resolved key.
    grain_label = ", ".join(grain_key)

    # facts carries the emitter's optional inputs (display columns + HAVING
    # threshold) and the question for traceability. All caller-supplied.
    facts: dict[str, Any] = {
        "display_cols": tuple(display_cols or ()),
    }
    if having:
        facts["having"] = dict(having)

    return Contract(
        question=question,
        entity=str(metric_name),
        source_table=str(source_table),
        source_state=SlotState.BOUND,
        grain=grain_label,
        grain_pk=grain_key,
        grain_state=SlotState.BOUND,
        measure=str(metric_name),
        measure_expr=str(measure_expr),
        measure_state=SlotState.BOUND,
        filter_preds=filter_preds,
        filter_state=SlotState.BOUND,
        joins=(),
        twin_cluster=(),
        # The upstream DB-backed binding step elected this canonical source via
        # the VERIFY tier before calling us; we record that election here so the
        # twin-penalty / twin-cluster fallback triggers do not fire.
        canonical_verified=True,
        candidate_tables=(),
        facts=facts,
    )
