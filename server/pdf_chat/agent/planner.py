"""Phase 3 planner/router — typed intent classification + bypass decision.

Mirrors the main system's ``app/services/semantic_planner.py`` contract:
a confidence-scored plan with a TYPED ``fallback_reason`` string and a typed
``intent``. High-confidence simple queries (or any cached query) BYPASS the
agentic tool loop entirely; everything else falls through to the loop with a
logged, typed reason.

Design rules (spec §3 invariants):
  * The planning model is chosen via ``model_router.select_model(task=
    QUERY_PLANNING, ...)`` — the planner never hardcodes a model id. The chosen
    ``ModelChoice`` is recorded on the result signals for the synthesis router.
  * NO magic literal: the bypass confidence floor resolves via
    ``get_tunable(container_id, PLANNER_BYPASS_CONFIDENCE, ...)`` and the
    decision is emitted via ``log_gate_decision`` so a score is never
    compared-and-discarded silently.
  * ``plan_query`` NEVER raises — a backend/parse failure degrades to a typed
    ``fallback_reason`` ("planner_error:<E>") with ``bypass=False`` so the loop
    is taken (the safe, fully-grounded path).
  * Intent is a stable INTENT-layer literal (``local|global|cross_domain|
    definitional``), never a learned customer-domain label; an out-of-vocab
    classification is coerced to the safe default ``"local"``.

Pure module — safe to import with zero infra. The LLM is injected (a
``PdfLlm``-like object exposing ``async generate(system, user, *, container_id,
signals)``); tests pass an in-memory ``FakeLlm``.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Literal

import structlog

from ..model_router import TaskClass, select_model
from ..tunables import get_tunable, log_gate_decision

_log = structlog.get_logger("pdf_chat.agent.planner")

# ── Typed intent vocabulary (stable INTENT layer; spec §3 invariant 6) ──────
QueryIntent = Literal["local", "global", "cross_domain", "definitional"]
_VALID_INTENTS: frozenset[str] = frozenset(
    {"local", "global", "cross_domain", "definitional"}
)
_DEFAULT_INTENT: QueryIntent = "local"

# ── Named tunable keys + defaults (single-source; integration registers these
# in TUNABLE_DEFAULTS — see this module's return notes). The named default is
# passed at every call site so the module stays import-safe pre-integration. ─
PLANNER_BYPASS_CONFIDENCE = "agent.planner_bypass_confidence"
PLANNER_BYPASS_CONFIDENCE_DEFAULT = 0.80

# Intents that are inherently multi-source / loop-requiring: even a confident
# classification of these does NOT bypass (they need the tool loop to gather
# evidence across documents / definitions).
_LOOP_FORCING_INTENTS: frozenset[str] = frozenset({"global", "cross_domain"})


@dataclass
class PlannerResult:
    """The plan: typed intent + confidence + bypass + typed fallback + signals."""

    intent: QueryIntent = _DEFAULT_INTENT
    confidence: float = 0.0
    bypass: bool = False
    fallback_reason: str | None = None  # typed: "low_confidence:<c>"|"planner_error:<E>"|"ambiguous_intent"
    signals: dict = field(default_factory=dict)  # {cross_domain, definitional, ...} for the router


_PLANNER_SYSTEM = (
    "You are a query planner for a document-retrieval agent. Classify the user "
    "query into exactly one intent and report your confidence.\n"
    "Intents:\n"
    "  - local: a single fact answerable from one passage/section.\n"
    "  - global: a summary requiring many passages across a document.\n"
    "  - cross_domain: requires comparing/joining evidence across multiple documents.\n"
    "  - definitional: asks for the meaning/definition of a term.\n"
    "Reply with ONLY a JSON object: "
    '{"intent": "<one of the four>", "confidence": <float 0..1>, '
    '"multi_part": <true if the query has multiple distinct sub-questions>}.'
)


def _parse_plan(raw: str) -> dict[str, Any]:
    """Extract the planner JSON. Tolerates code fences / surrounding prose.

    Raises ``ValueError`` if no JSON object with an ``intent`` is found — the
    caller turns that into a typed ``planner_error`` fallback.
    """
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in planner reply")
    obj = json.loads(match.group(0))
    if not isinstance(obj, dict) or "intent" not in obj:
        raise ValueError("planner JSON missing 'intent'")
    return obj


def _coerce_intent(raw_intent: Any) -> QueryIntent:
    """Map the LLM intent to a typed literal; unknown → safe default ``local``."""
    val = str(raw_intent or "").strip().lower()
    return val if val in _VALID_INTENTS else _DEFAULT_INTENT  # type: ignore[return-value]


def _coerce_confidence(raw_conf: Any) -> float:
    try:
        conf = float(raw_conf)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, conf))


async def plan_query(
    query: str,
    *,
    container_id: str,
    llm,
    cached: bool = False,
) -> PlannerResult:
    """Classify ``query`` into a typed intent + decide whether to bypass the loop.

    Flow:
      1. Pick the planning model via ``select_model(task=QUERY_PLANNING)`` (the
         router stays the single model seam; the choice is recorded on signals).
      2. Prompt the LLM, parse intent + confidence (never raises on failure).
      3. Build router signals (cross_domain / definitional booleans the
         downstream synthesis router reads).
      4. Compare confidence to the tunable bypass floor via ``log_gate_decision``;
         a cached query bypasses unconditionally; loop-forcing intents never
         bypass. A below-floor confidence yields a typed ``low_confidence:<c>``
         fallback.
    """
    result = PlannerResult()

    # 1. Model selection seam — the planner never hardcodes a model id. Signals
    #    here are pre-classification, so they carry no intent flags yet; the
    #    router can still apply confidence/figure escalation policy.
    try:
        choice = select_model(
            task=TaskClass.QUERY_PLANNING,
            container_id=container_id,
            signals={},
        )
        result.signals["planning_model"] = choice.model_id
    except Exception as exc:  # pragma: no cover - router is pure; defensive only
        _log.warning("pdf_chat.planner.select_model_failed", error=repr(exc))

    # 2. Classify (never raises).
    try:
        raw = await llm.generate(
            _PLANNER_SYSTEM,
            query,
            container_id=container_id,
            signals=result.signals,
        )
        plan = _parse_plan(raw)
    except Exception as exc:
        # Typed fallback mirroring semantic_planner's "planner_error:<E>" style.
        result.fallback_reason = f"planner_error:{type(exc).__name__}"
        result.bypass = False
        log_gate_decision(
            "agent.planner_bypass",
            score=0.0,
            threshold=get_tunable(
                container_id, PLANNER_BYPASS_CONFIDENCE, PLANNER_BYPASS_CONFIDENCE_DEFAULT
            ),
            outcome="loop",
            container_id=container_id,
            fallback_reason=result.fallback_reason,
        )
        return result

    # 3. Typed intent + confidence + router signals.
    result.intent = _coerce_intent(plan.get("intent"))
    result.confidence = _coerce_confidence(plan.get("confidence"))
    multi_part = bool(plan.get("multi_part", False))
    result.signals["cross_domain"] = result.intent == "cross_domain"
    result.signals["definitional"] = result.intent == "definitional"
    result.signals["multi_part"] = multi_part

    floor = get_tunable(
        container_id, PLANNER_BYPASS_CONFIDENCE, PLANNER_BYPASS_CONFIDENCE_DEFAULT
    )

    # 4a. Cached query → bypass unconditionally (the answer is reused; no loop).
    if cached:
        result.bypass = True
        result.fallback_reason = None
        log_gate_decision(
            "agent.planner_bypass",
            score=result.confidence,
            threshold=floor,
            outcome="bypass",
            container_id=container_id,
            intent=result.intent,
            reason="cached",
        )
        return result

    # 4b. Loop-forcing intents (global/cross_domain) or multi-part asks always
    #     take the loop — they require multi-source evidence gathering.
    if result.intent in _LOOP_FORCING_INTENTS or multi_part:
        result.bypass = False
        result.fallback_reason = (
            "ambiguous_intent" if multi_part else f"intent_requires_loop:{result.intent}"
        )
        log_gate_decision(
            "agent.planner_bypass",
            score=result.confidence,
            threshold=floor,
            outcome="loop",
            container_id=container_id,
            intent=result.intent,
            fallback_reason=result.fallback_reason,
        )
        return result

    # 4c. Confidence gate for simple (local/definitional) queries.
    if result.confidence >= floor:
        result.bypass = True
        result.fallback_reason = None
        outcome = "bypass"
    else:
        result.bypass = False
        result.fallback_reason = f"low_confidence:{round(result.confidence, 2)}"
        outcome = "loop"

    log_gate_decision(
        "agent.planner_bypass",
        score=result.confidence,
        threshold=floor,
        outcome=outcome,
        container_id=container_id,
        intent=result.intent,
        fallback_reason=result.fallback_reason,
    )
    return result
