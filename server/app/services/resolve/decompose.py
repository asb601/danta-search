"""Query-time question decomposition for the brain-resolve seam.

ONE gpt-4o-mini call that turns the raw natural-language question into a typed,
structured intent the coordinator can act on: which business ENTITIES the answer
spans (one retrieval pass per entity), what the INTENT is, the MEASURE concept,
the GRAIN (per-entity vs per-period), and any FILTERS / THRESHOLD / time grain.

This module is the FRONT of the brain seam. It does NOT pick tables or columns
(that is the brain, over the retrieved slice) and it does NOT compute anything.
It is purely the structured restatement of the question — the entities it returns
drive ``search_per_entity``; the filters/threshold/time_grain it returns are
CARRIED for downstream use but are deliberately NOT used to match tables (the
brain value-verifies those against real evidence).

Design properties (enforced):
  * Evidence-free. Reads only the question (+ an optional planner intent seed);
    no DB, no catalog, no cross-file conclusions.
  * No hardcoded business terms. Every entity/measure/filter comes from the
    question or the planner seed — there are no name lists or ERP dictionaries.
  * Never raises. Any LLM / parse / shape error → ``None`` so the caller falls
    through to the existing path. The seam is strictly additive.
"""
from __future__ import annotations

import asyncio

import structlog

from app.core.config import get_settings
from app.core.llm_tasks import safe_parse_json
from app.core.openai_client import get_client

logger = structlog.get_logger("resolve.decompose")

# Allowed enumerations. These are the JSON GRAMMAR of the contract slot, not
# business knobs — the grain/op/time-grain vocabulary the runtime understands.
_GRAINS: frozenset[str] = frozenset({"entity", "time"})
_TIME_GRAINS: frozenset[str] = frozenset({"month", "quarter", "year"})
_MAX_ENTITIES = 6  # narrow fan-out: one retrieval + one brain pass per entity


def _norm_entity(raw: object) -> str:
    """Lowercase, whitespace-collapsed string form of an entity token. The LLM
    may emit a bare string or a small object; accept either without inventing
    semantics. Pure."""
    if isinstance(raw, dict):
        for key in ("entity", "name", "concept", "label"):
            value = raw.get(key)
            if str(value or "").strip():
                return str(value).strip()
        return ""
    return str(raw or "").strip()


def _seed_block(intent_seed: dict | None) -> str:
    """Render the optional planner intent seed (ctx['intent_plan']) into a short
    hint block. The seed is advisory — the LLM may extend or override it — so it
    only nudges entity selection, never constrains it."""
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
    return "\nA prior lightweight planner extracted the following — use it as a hint, " \
           "extend or correct it as needed:\n" + "\n".join(lines)


def _prompt(question: str, intent_seed: dict | None, as_of: object) -> str:
    as_of_line = ""
    if as_of:
        as_of_line = (f"\nThe data's latest available date (as-of anchor) is {as_of}; "
                      f"resolve any relative period against it when you set time_grain.")
    return f"""You are the question-decomposition stage of a business analytics agent.
Restate the question as a typed JSON intent. You DO NOT pick tables or columns and
you DO NOT compute anything — you name the business ENTITIES the answer spans (so
each can be retrieved separately), the INTENT, the MEASURE concept, the GRAIN, and
any filters / threshold / time grain present in the question.{as_of_line}{_seed_block(intent_seed)}

QUESTION: {question}

Return ONLY JSON:
{{
  "entities": ["snake_case_business_object"],   // 1-{_MAX_ENTITIES} objects the answer spans (e.g. customer, invoice, vendor). Expand abbreviations (PO -> purchase_order).
  "intent": "short action+domain slug (e.g. aggregation, open_items, top_n_lookup, detail_lookup)",
  "measure_concept": "the metric in business words, or null (e.g. open receivables, cash received, total amount)",
  "grain": "entity" | "time" | null,            // entity = one row per <id>; time = per period; null if not implied
  "grain_entity": "the entity the grain is per, or null (e.g. customer for 'by customer')",
  "filters": [{{"concept": "the dimension in words", "op": "=", "value": "..."}}],  // values/dimensions named in the question; [] if none
  "threshold": {{"op": ">", "value": 500000}} | null,   // a per-group cutoff ('over 500k'), else null
  "time_grain": "month" | "quarter" | "year" | null     // only if the question asks 'by month/quarter/year'
}}
Rules:
1. Entities are data objects that need separate retrieval — NOT display fields,
   time ranges, metrics, or output instructions.
2. Prefer breadth: a question spanning two objects (e.g. customer AND vendor)
   yields two entities, not one.
3. measure_concept is the business term for the number being asked for; leave it
   null for a pure detail/listing question.
4. Only set time_grain when the question explicitly buckets by a period."""


async def decompose_question(
    question: str,
    intent_seed: dict | None = None,
    as_of: object | None = None,
) -> dict | None:
    """Decompose a question into a typed intent dict, or ``None`` on any failure.

    ``intent_seed`` is the planner's ``ctx['intent_plan']`` (a dict, or a
    BusinessIntentPlan-like object exposing ``to_dict``) used only as a hint. The
    returned dict carries ``filters`` / ``threshold`` / ``time_grain`` for
    downstream use, but those are NOT used to match tables — the brain
    value-verifies the real columns. Never raises.
    """
    if not question or not question.strip():
        return None

    # Normalise the seed to a plain dict without coupling to the planner type.
    seed: dict | None = None
    if intent_seed is not None:
        if hasattr(intent_seed, "to_dict"):
            try:
                seed = intent_seed.to_dict()
            except Exception:  # noqa: BLE001 — a misbehaving seed must never break decompose
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
            max_completion_tokens=500,
        )
        return safe_parse_json((resp.choices[0].message.content or "{}").strip())

    try:
        parsed = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001 — never raise; caller falls through
        logger.warning("decompose_error", error=str(exc)[:200])
        return None

    if not isinstance(parsed, dict):
        logger.info("decompose_bad_json")
        return None

    # Normalise entities — dedup, drop blanks, cap the fan-out. The brain, not
    # this stage, decides whether each entity is answerable.
    entities: list[str] = []
    seen: set[str] = set()
    raw_entities = parsed.get("entities")
    if not isinstance(raw_entities, list):
        raw_entities = [raw_entities] if raw_entities else []
    for raw in raw_entities:
        ent = _norm_entity(raw)
        key = ent.lower()
        if ent and key not in seen:
            entities.append(ent)
            seen.add(key)
    entities = entities[:_MAX_ENTITIES]
    if not entities:
        logger.info("decompose_no_entities", question=question[:160])
        return None

    grain = parsed.get("grain")
    if grain not in _GRAINS:
        grain = None
    time_grain = parsed.get("time_grain")
    if time_grain not in _TIME_GRAINS:
        time_grain = None

    filters = parsed.get("filters")
    if not isinstance(filters, list):
        filters = []
    threshold = parsed.get("threshold")
    if not isinstance(threshold, dict):
        threshold = None

    out = {
        "entities": entities,
        "intent": str(parsed.get("intent") or "").strip() or "unknown",
        "measure_concept": (str(parsed.get("measure_concept")).strip()
                            if parsed.get("measure_concept") else None),
        "grain": grain,
        "grain_entity": (str(parsed.get("grain_entity")).strip()
                         if parsed.get("grain_entity") else None),
        "filters": filters,
        "threshold": threshold,
        "time_grain": time_grain,
    }
    logger.info(
        "decompose_ok",
        entities=entities,
        intent=out["intent"],
        measure=out["measure_concept"],
        grain=out["grain"],
        time_grain=out["time_grain"],
    )
    return out
