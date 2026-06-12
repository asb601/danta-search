"""[3c] PROPOSE — one mini call that fills the typed contract slots for ONE step.

Given the question, the step's full intent (entity + measure_concept + grain), and
the per-file EVIDENCE assembled for the step's candidate slice, this makes ONE
gpt-4o-mini call (temp 0, the SAME client path planner.py uses) and maps the
returned JSON into a typed ``ProposedContract``. It does the SEMANTIC judgments the
brain is good at — which table is canonical for THIS step, what one output row
means (grain), which column is the measure, what to filter — and nothing else.

CRITICAL (INVARIANTS I1/I2): mini fills ONLY the typed slots, choosing
table/column/agg/filter names VERBATIM from the evidence. It NEVER writes SQL,
invents an identifier, or computes a number. A ``ProposedContract`` carries no SQL
string — rendering is a separate, deterministic stage (renderer.py).

The prompt + the answerability gate are LIFTED from ``resolve.brain._evidence_prompt``
/ ``propose_contract`` (the navigator is self-contained — it does not import from
``app.services.resolve.*``). What is added: the step's intent (entity, measure
concept, grain) is rendered into the prompt so mini picks for THIS sub-intent, not
a re-read of the whole question.

Robustness (lifted from brain.propose_contract): never raises. An empty packet, an
LLM error, bad JSON, or ``answerable=false`` all yield ``None`` so the driver can
re-propose once or abstain (I6/I12).
"""
from __future__ import annotations

import asyncio

import structlog

from app.core.config import get_settings
from app.core.llm_tasks import safe_parse_json
from app.core.openai_client import get_client
from app.services.navigator.types import (
    EvidencePacket,
    IntentStep,
    ProposedContract,
)

logger = structlog.get_logger("navigator.proposer")

# Bounds — match the legacy evidence prompt so cost stays flat regardless of slice
# width. These are size guards / the grammar mini chooses among, not business knobs.
_SAMPLE_ROWS = 3


def _intent_block(step: IntentStep | None) -> str:
    """Render the step's full intent (entity + measure concept + grain) into a
    short context block so mini fills the contract for THIS sub-intent. Pure."""
    if step is None:
        return ""
    bits: list[str] = []
    if step.entity:
        bits.append(f"business object = {step.entity}")
    if step.measure_concept:
        bits.append(f"measure concept = {step.measure_concept}")
    if step.grain == "time" and step.time_grain:
        bits.append(f"grain = per {step.time_grain} (time)")
    elif step.grain == "entity":
        per = step.grain_entity or step.entity
        bits.append(f"grain = per {per} (entity)" if per else "grain = entity")
    if not bits:
        return ""
    return (
        "\nTHIS STEP'S INTENT (resolve the contract for THIS sub-intent, not the "
        "whole question): " + "; ".join(bits)
    )


def _correction_block(prior_failure: str | None) -> str:
    """A corrective directive for the SECOND propose attempt (FIX C / L7): render the
    first attempt's verify-failure reason so mini picks DIFFERENTLY instead of
    repeating the same losing pick at temp 0. Empty for the first attempt (None) so
    the first-attempt prompt is byte-identical to today's. Pure."""
    if not prior_failure:
        return ""
    return (
        f"\nYour previous choice FAILED verification: {prior_failure}. Choose a "
        f"different table/column/grain/filter that passes; do not repeat the same pick."
    )


def _evidence_prompt(
    question: str, step: IntentStep | None, evidence: list[dict],
    time_window: tuple | None, prior_failure: str | None = None,
) -> str:
    """Build the BRAIN prompt. LIFTED verbatim from ``resolve.brain._evidence_prompt``
    with the step-intent block added so mini picks for THIS step. Pure.

    ``prior_failure`` (FIX C / L7): when set, a correction directive is rendered so
    a re-propose self-corrects. None ⇒ no directive (first-attempt prompt unchanged).
    """
    # <seam: KB-build also adds a separate "declared-measures" line inside this fn>
    lines: list[str] = []
    for e in evidence:
        lines.append(f"TABLE {e['table']}  (rows={e['row_count']:,}, coverage {e['coverage']})")
        # Business-context discriminator — rendered ONLY when present, so the prompt
        # is unchanged for files without a reliable classification.
        ctx_bits = []
        if e.get("polarity"):
            ctx_bits.append(f"ledger_side={e['polarity']}")
        if e.get("process_role"):
            ctx_bits.append(f"process_role={e['process_role']}")
        if e.get("erp_module"):
            ctx_bits.append(f"module={e['erp_module']}")
        if ctx_bits:
            lines.append("  business_context: " + ", ".join(ctx_bits))
        lines.append(f"  purpose(good_for): {e.get('good_for')}")
        lines.append(f"  desc: {e.get('description')}")
        col_strs = [f"{c['name']}:{c['type'] or '?'}" + (f"[{c['role']}]" if c['role'] else "")
                    for c in e.get("columns", [])]
        lines.append("  columns: " + ", ".join(col_strs))
        sample = (e.get("sample_rows") or [])[:_SAMPLE_ROWS]
        if sample:
            lines.append(f"  sample rows: {sample}")
        lines.append("")
    tw = ""
    if time_window:
        tw = (f"\nA relative time window was detected for this question: "
              f"{time_window[0]} to {time_window[1]} (already resolved against the "
              f"data's latest date — use this if the question is time-scoped; pick the "
              f"date column to apply it to).")
    return f"""You are the analytical BRAIN of a data agent. You DO NOT write SQL. You read the
EVIDENCE for a few candidate tables and decide how to answer the question by filling
a typed contract. Pick the SINGLE best table from the evidence (read the sample rows
and column roles — these are ERP tables and several look alike). Choose the grain
(what one output row represents), the measure column + aggregation, and any filters.
Use ONLY table names and column names that appear verbatim in the evidence.{_intent_block(step)}{tw}{_correction_block(prior_failure)}

QUESTION: {question}

EVIDENCE:
{chr(10).join(lines)}
Return ONLY JSON:
{{
  "answerable": true|false,            // false if no table here fits the question
  "table": "EXACT_TABLE_NAME",
  "table_reason": "one line: why this table over the others",
  "grain": "entity" | "time",          // entity = one row per <id>; time = per period
  "grain_column": "EXACT_COLUMN",      // the id column (entity) or the date column (time)
  "time_bucket": "month"|"quarter"|"year"|null,   // only for grain=time
  "measure_column": "EXACT_COLUMN",
  "measure_agg": "SUM"|"COUNT"|"AVG"|"MAX"|"MIN"|"COUNT_DISTINCT",
  "filters": [{{"column":"EXACT_COLUMN","op":"=","value":"..."}}],  // e.g. open status; [] if none
  "time_filter_column": "EXACT_DATE_COLUMN"|null,  // the date column the time window applies to
  "having": {{"op":">","value":500000}}|null,      // per-group threshold, else null
  "top_n": 20|null,
  "order": "desc"|"asc"
}}
Rules: choose the measure that matches the business term (e.g. cash received → the
applied/received amount, not the original invoice amount — use roles + sample values
to decide). If the question asks "by month/quarter/year", grain=time. If it asks for
customers/vendors/etc, grain=entity with that id as grain_column. Abstain
(answerable=false) rather than force a bad table.
IDENTITY: several tables here may share nearly identical schemas (lookalikes). Two
tables with DIFFERENT business_context ledger sides are NOT lookalikes — they have
opposite meaning and must never be conflated; pick the one whose ledger side and
process role match what the question is about. Row count and coverage break ties ONLY
within the SAME ledger side and process role — never across sides. Among same-side
lookalikes, when the question wants the GENERAL fact prefer the CANONICAL MASTER (the
most complete/granular table — usually the largest row count and broadest coverage),
NOT a narrower view (a history/archive/delinquency/interim subset). Only choose a
subset table when the question explicitly asks for that subset."""


def _to_contract(step_id: str, out: dict) -> ProposedContract | None:
    """Map mini's raw JSON answer into a typed ``ProposedContract`` — typed slots
    only, NO SQL. Filters are kept as the raw verbatim {column,op,value} dicts; the
    verifier value-checks and normalises them. Pure."""
    table = str(out.get("table") or "").strip()
    if not table:
        return None
    raw_filters = out.get("filters")
    filters: tuple = tuple(f for f in raw_filters if isinstance(f, dict)) if isinstance(
        raw_filters, list
    ) else ()
    top_n = out.get("top_n") if isinstance(out.get("top_n"), int) else None
    having = out.get("having") if isinstance(out.get("having"), dict) else None
    return ProposedContract(
        step_id=step_id,
        table=table,
        table_reason=str(out.get("table_reason") or "")[:200] or None,
        grain_kind=str(out.get("grain")) if out.get("grain") in ("entity", "time") else None,
        grain_column=str(out.get("grain_column")) if out.get("grain_column") else None,
        time_bucket=str(out.get("time_bucket")) if out.get("time_bucket") else None,
        measure_column=str(out.get("measure_column")) if out.get("measure_column") else None,
        measure_agg=str(out.get("measure_agg")).upper() if out.get("measure_agg") else None,
        filters=filters,
        time_filter_column=str(out.get("time_filter_column")) if out.get("time_filter_column") else None,
        having=having,
        top_n=top_n,
        order=str(out.get("order") or "desc").lower(),
    )


async def propose(
    question: str,
    step: IntentStep,
    ev: EvidencePacket,
    time_window: tuple | None = None,
    prior_failure: str | None = None,
) -> ProposedContract | None:
    """ONE mini call → a typed ``ProposedContract`` for ``step``, or ``None``.

    Returns ``None`` on an empty slice, an LLM error, bad JSON, or
    ``answerable=false`` (abstain). NEVER raises — the driver re-proposes once or
    abstains (I6/I12). mini fills ONLY typed slots from the evidence's verbatim
    names; it never writes SQL or computes (I1/I2).

    ``prior_failure`` (FIX C / L7): the first attempt's verify-failure reason. When
    the driver re-proposes after a verify failure it passes this so mini self-corrects
    (picks DIFFERENTLY) instead of repeating the same losing pick at temp 0. None on
    the first attempt ⇒ the prompt is byte-identical to today's.
    """
    files = list(ev.files) if ev is not None else []
    if not files:
        return None
    step_id = step.step_id if step is not None else (ev.step_id if ev is not None else "")

    def _run() -> dict:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user",
                       "content": _evidence_prompt(question, step, files, time_window,
                                                   prior_failure)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=600,
        )
        return safe_parse_json((resp.choices[0].message.content or "{}").strip())

    try:
        out = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001 — never raise; driver re-proposes/abstains
        logger.warning("propose_llm_error", error=str(exc)[:200])
        return None

    if not isinstance(out, dict) or not out.get("answerable"):
        logger.info(
            "propose_abstain",
            reason=str(out.get("table_reason", ""))[:160] if isinstance(out, dict) else "bad_json",
        )
        return None

    pc = _to_contract(step_id, out)
    if pc is None:
        logger.info("propose_no_table")
        return None
    logger.info(
        "propose_ok",
        step_id=pc.step_id,
        table=pc.table,
        measure=f"{pc.measure_agg}({pc.measure_column})",
        grain=pc.grain_kind,
    )
    return pc
