"""[1] PLAN — decompose a question into an ORDERED DAG of intent-steps.

ONE gpt-4o-mini call (temp 0) over the SAME client path ``decompose.py`` uses
(``app.core.openai_client.get_client`` -> ``client.chat.completions.create`` with
``AZURE_OPENAI_DEPLOYMENT_MINI``). The discipline (never-raises, ``safe_parse_json``,
caps, dedup) is lifted from ``decompose.py``.

What changes vs the legacy decomposer: the output unit is an INTENT-STEP, not a
bare entity token (INVARIANT I3). mini emits an ordered DAG where each step is a
FULL sub-intent that resolves to exactly one table downstream. Examples:

    "total vendor spend"
        -> 1 LOOKUP step.
    "customer receipts vs vendor payments"
        -> 2 LOOKUP steps (distinct entities, no depends_on).
    "ratio of overdue to total invoices per vendor"
        -> 2 LOOKUP steps + 1 COMPOSE step whose depends_on = the two LOOKUP
           step_ids and compose_expr = {op:"ratio", left_step, right_step}.

CRITICAL (INVARIANTS I1/I2): mini does ONLY planning here. It must NOT name
tables, columns, or SQL, and must NOT compute anything. Table/column selection
happens later (PROPOSE/VERIFY). ``entity`` is a business-object concept
(snake_case); ``measure_concept`` is a business term — both resolved to real
tables/columns downstream, never in this module.

Robustness (lifted from decompose.py): never raises -> returns ``None`` on any
error / empty / bad-JSON. Guards: cap steps (``_MAX_STEPS``), dedup, and REJECT a
DAG with a cycle in ``depends_on`` or a dangling ``depends_on`` / ``compose_expr``
reference -> ``None``.
"""
from __future__ import annotations

import asyncio

import structlog

from app.core.config import get_settings
from app.core.llm_tasks import safe_parse_json
from app.core.openai_client import get_client
from app.services.navigator.types import IntentStep, StepDAG, StepKind

logger = structlog.get_logger("navigator.planner")

# Grammar of the plan slots (the vocabulary the runtime understands), NOT
# business knobs. mini chooses among these enumerations.
_KINDS: dict[str, StepKind] = {k.value: k for k in StepKind}
_GRAINS: frozenset[str] = frozenset({"entity", "time"})
_TIME_GRAINS: frozenset[str] = frozenset({"month", "quarter", "year"})
_COMPOSE_OPS: frozenset[str] = frozenset({"ratio", "diff", "growth", "share"})

_MAX_STEPS = 6  # narrow fan-out: one resolution pass per step


def _seed_block(intent_seed: dict | None) -> str:
    """Render the optional planner intent seed into a short advisory hint block.
    The seed only nudges step selection; mini may extend or correct it. Pure."""
    if not intent_seed:
        return ""
    entities = intent_seed.get("entities") or []
    constraints = intent_seed.get("constraints") or {}
    lines: list[str] = []
    if entities:
        lines.append("  entities (from planner): " + ", ".join(str(e) for e in entities))
    if constraints:
        lines.append(f"  constraints (from planner): {constraints}")
    if not lines:
        return ""
    return (
        "\nA prior lightweight planner extracted the following — use it as a hint, "
        "extend or correct it as needed:\n" + "\n".join(lines)
    )


def _prompt(question: str, intent_seed: dict | None, as_of: object) -> str:
    as_of_line = ""
    if as_of:
        as_of_line = (
            f"\nThe data's latest available date (as-of anchor) is {as_of}; "
            f"resolve any relative period against it when you set time_grain."
        )
    return f"""You are the PLAN stage of a business analytics agent.
Decompose the question into an ORDERED DAG of INTENT-STEPS. Each step is a FULL
sub-intent that will later resolve to exactly ONE table. You DO NOT name tables,
columns, or SQL, and you DO NOT compute anything — table/column selection and all
arithmetic happen in later, separate stages.{as_of_line}{_seed_block(intent_seed)}

QUESTION: {question}

Return ONLY JSON:
{{
  "steps": [
    {{
      "step_id": "s1",                         // unique short id; later steps reference earlier ids
      "kind": "LOOKUP" | "JOIN" | "COMPOSE",   // LOOKUP=one sub-intent->one table; JOIN=stitch tables; COMPOSE=cross-step math only
      "entity": "snake_case_business_object",  // the business object this step is about (e.g. vendor, customer, invoice); null for COMPOSE
      "measure_concept": "the metric in business words, or null (e.g. total spend, overdue invoices)",
      "grain": "entity" | "time" | null,       // entity = one row per <id>; time = per period; null if not implied
      "grain_entity": "the entity the grain is per, or null (e.g. vendor for 'per vendor')",
      "time_grain": "month" | "quarter" | "year" | null,
      "filters": [{{"concept": "dimension in words", "op": "=", "value": "..."}}],  // [] if none
      "threshold": {{"op": ">", "value": 500000}} | null,
      "depends_on": ["s1"],                    // step_ids this step needs first; [] for independent LOOKUPs
      "join_entities": ["entity_a", "entity_b"],  // for JOIN steps, the entities being joined; [] otherwise
      "compose_expr": {{"op": "ratio", "left_step": "s1", "right_step": "s2"}}  // ONLY for COMPOSE: op in ratio|diff|growth|share
    }}
  ]
}}
Rules:
1. "Measure BY/PER a dimension" is ONE LOOKUP step, NOT two. The grouping dimension
   is NOT a separate business object. For "invoice amount by vendor", "salary by
   department", "total spend per supplier", "orders by customer", "top N suppliers
   by invoice amount": emit a SINGLE LOOKUP with entity = the object that OWNS the
   measure (invoice, salary, order — NOT the grouping dimension), grain = "entity",
   grain_entity = the dimension (vendor, department, customer). Do NOT create a
   second LOOKUP and do NOT create a JOIN just to group by a dimension.
2. Use SEPARATE LOOKUP steps (plus a JOIN or COMPOSE) ONLY when the answer truly
   needs data from two DIFFERENT business objects — e.g. "customer receipts vs
   vendor payments" (compare two objects) or "orders AND their shipments" (stitch
   two objects). When in doubt, prefer ONE LOOKUP — over-splitting causes the whole
   query to fail.
3. A cross-step calculation (e.g. "ratio of X to Y", "growth", "share of") is a
   separate COMPOSE step. Its depends_on lists the LOOKUP step_ids it combines and
   its compose_expr names the op and the left/right step_ids. COMPOSE has no entity
   and never names a table.
4. NEVER output a table name, a column name, or SQL. entity and measure_concept are
   business concepts only.
5. Emit at most {_MAX_STEPS} steps. Keep step_ids unique. Only reference step_ids that exist."""


def _norm_str(value: object) -> str | None:
    s = str(value).strip() if value is not None else ""
    return s or None


def _norm_id_list(raw: object) -> list[str]:
    """Coerce a depends_on / id list into a clean list of non-blank str ids."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in raw:
        sid = str(item).strip() if item is not None else ""
        if sid and sid not in seen:
            out.append(sid)
            seen.add(sid)
    return out


def _norm_compose_expr(raw: object) -> dict | None:
    """Validate a compose_expr: must be a dict with a known op and two step refs."""
    if not isinstance(raw, dict):
        return None
    op = str(raw.get("op") or "").strip().lower()
    if op not in _COMPOSE_OPS:
        return None
    left = _norm_str(raw.get("left_step"))
    right = _norm_str(raw.get("right_step"))
    if not left or not right:
        return None
    return {"op": op, "left_step": left, "right_step": right}


def _build_step(raw: object) -> IntentStep | None:
    """Normalise one raw step dict into an IntentStep, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    step_id = _norm_str(raw.get("step_id"))
    if not step_id:
        return None
    kind = _KINDS.get(str(raw.get("kind") or "").strip().upper())
    if kind is None:
        return None

    grain = raw.get("grain")
    if grain not in _GRAINS:
        grain = None
    time_grain = raw.get("time_grain")
    if time_grain not in _TIME_GRAINS:
        time_grain = None

    filters = raw.get("filters")
    filters_t: tuple = tuple(filters) if isinstance(filters, list) else ()
    threshold = raw.get("threshold")
    if not isinstance(threshold, dict):
        threshold = None

    compose_expr = _norm_compose_expr(raw.get("compose_expr"))

    return IntentStep(
        step_id=step_id,
        kind=kind,
        entity=_norm_str(raw.get("entity")),
        measure_concept=_norm_str(raw.get("measure_concept")),
        grain=grain,
        grain_entity=_norm_str(raw.get("grain_entity")),
        time_grain=time_grain,
        filters=filters_t,
        threshold=threshold,
        depends_on=tuple(_norm_id_list(raw.get("depends_on"))),
        join_entities=tuple(_norm_id_list(raw.get("join_entities"))),
        compose_expr=compose_expr,
    )


def _has_cycle(steps: list[IntentStep]) -> bool:
    """True if the depends_on graph contains a cycle. Iterative DFS with colors."""
    adj: dict[str, tuple[str, ...]] = {s.step_id: s.depends_on for s in steps}
    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {sid: WHITE for sid in adj}

    for root in adj:
        if color[root] != WHITE:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                color[node] = BLACK
                continue
            if color[node] == GRAY:
                continue
            color[node] = GRAY
            stack.append((node, True))
            for dep in adj.get(node, ()):  # dep must exist (dangling checked elsewhere)
                if color.get(dep, BLACK) == GRAY:
                    return True
                if color.get(dep, BLACK) == WHITE:
                    stack.append((dep, False))
    return False


def _refs_valid(steps: list[IntentStep], ids: set[str]) -> bool:
    """Every depends_on id and every compose_expr step ref must point at a real
    step in this DAG. A dangling reference rejects the whole plan."""
    for step in steps:
        for dep in step.depends_on:
            if dep not in ids:
                return False
        if step.compose_expr is not None:
            for key in ("left_step", "right_step"):
                ref = step.compose_expr.get(key)
                if ref not in ids:
                    return False
    return True


async def plan(
    question: str,
    intent_seed: dict | None = None,
    as_of: object | None = None,
) -> StepDAG | None:
    """Decompose ``question`` into a validated ``StepDAG``, or ``None`` on any
    failure. Never raises (mirrors decompose.py): any LLM / parse / shape error,
    an empty plan, a cycle, or a dangling reference all yield ``None`` so the
    driver can abstain or fall through.
    """
    if not question or not question.strip():
        return None

    # Normalise the seed to a plain dict without coupling to the planner type.
    seed: dict | None = None
    if intent_seed is not None:
        if hasattr(intent_seed, "to_dict"):
            try:
                seed = intent_seed.to_dict()
            except Exception:  # noqa: BLE001 — a misbehaving seed must never break plan
                seed = None
        elif isinstance(intent_seed, dict):
            seed = intent_seed

    def _run() -> dict:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": _prompt(question, seed, as_of)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=700,
        )
        return safe_parse_json((resp.choices[0].message.content or "{}").strip())

    try:
        parsed = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001 — never raise; caller abstains/falls through
        logger.warning("plan_error", error=str(exc)[:200])
        return None

    if not isinstance(parsed, dict):
        logger.info("plan_bad_json")
        return None

    raw_steps = parsed.get("steps")
    if not isinstance(raw_steps, list) or not raw_steps:
        logger.info("plan_no_steps", question=question[:160])
        return None

    # Build + dedup by step_id (first occurrence wins), cap the fan-out.
    steps: list[IntentStep] = []
    seen_ids: set[str] = set()
    for raw in raw_steps:
        step = _build_step(raw)
        if step is None:
            continue
        if step.step_id in seen_ids:
            continue
        steps.append(step)
        seen_ids.add(step.step_id)
        if len(steps) >= _MAX_STEPS:
            break

    if not steps:
        logger.info("plan_no_valid_steps", question=question[:160])
        return None

    ids = {s.step_id for s in steps}
    if not _refs_valid(steps, ids):
        logger.info("plan_dangling_ref", question=question[:160])
        return None
    if _has_cycle(steps):
        logger.info("plan_cycle", question=question[:160])
        return None

    dag = StepDAG(question=question, steps=tuple(steps), intent=seed)
    logger.info(
        "plan_ok",
        n_steps=len(steps),
        kinds=[s.kind.value for s in steps],
        entities=[s.entity for s in steps],
    )
    return dag
