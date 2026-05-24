"""Business Intent Planner — lightweight structured query classifier.

Sits between query arrival and retrieve_with_scores(). Answers ONE question:
  "What is this query trying to do, and over what business entities?"

Responsibilities (strictly bounded):
  - Detect query BEHAVIOR  (aggregation, time filter, open-item, top-n, …)
  - Extract business ENTITIES  (the nouns the user is asking about)
  - Extract query CONSTRAINTS  (concrete values: date ranges, top-n counts)
  - Classify a single INTENT slug

NOT responsible for:
  - Resolving where entities live (→ graph resolver)
  - Matching semantic roles (→ retrieval layer)
  - Schema inspection or DB reads
  - SQL generation or join planning

Separation of concerns:
  PLANNER         → WHAT the user wants
  GRAPH RESOLVER  → WHERE concepts live in the schema
  SQL GENERATOR   → HOW to execute

Design constraints (non-negotiable):
  - Signal detection (behaviors, constraints, intent): deterministic regex. Zero LLM cost.
  - Entity extraction: one tiny GPT-4o-mini call per query. Semantic normalization only.
    Input = raw query. Output = strict JSON {"entities": [...]}. No chain-of-thought.
  - No DB reads. Output is a small, stable dataclass. Never raises.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.llm_tasks import extract_entities_for_query
from app.core.logger import chat_logger


# ── Behavior signal patterns ───────────────────────────────────────────────────

_AGG_RE = re.compile(
    r"\btotal\b|\bsum\b|\bcount\b|\bhow many\b|\bnumber of\b"
    r"|\baverage\b|\bavg\b|\bmaximum\b|\bmax\b|\bminimum\b|\bmin\b"
    r"|\bhighest\b|\blowest\b|\blargest\b|\bsmallest\b",
    re.I,
)

_TIME_RE = re.compile(
    r"\blast\s+\d+\s+days?\b|\blast\s+\d+\s+months?\b|\blast\s+month\b"
    r"|\blast\s+year\b|\bthis\s+month\b|\bthis\s+year\b|\bprevious\s+year\b"
    r"|\bQ[1-4]\s*20\d{2}\b|\b20\d{2}\b"
    r"|\bjanuary\b|\bfebruary\b|\bmarch\b|\bapril\b|\bjune\b|\bjuly\b"
    r"|\baugust\b|\bseptember\b|\boctober\b|\bnovember\b|\bdecember\b",
    re.I,
)

_OPEN_ITEM_RE = re.compile(
    r"\bopen\b|\bpending\b|\buncleared\b|\boverdue\b"
    r"|\boutstanding\b|\bnot cleared\b|\bnot paid\b|\bunpaid\b",
    re.I,
)

_DETAIL_RE = re.compile(
    r"\bshow\b|\blist\b|\bdisplay\b|\bfetch\b|\bgive me\b"
    r"|\brows\b|\brecords\b|\bdetails?\b|\bline items?\b",
    re.I,
)

_TOP_N_RE = re.compile(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b", re.I)

_COMPLEX_RE = re.compile(
    r"\bcompare\b|\btrend\b|\bforecast\b|\bcorrelation\b"
    r"|\bversus\b|\bvs\.?\b|\byoy\b|\bmom\b|\bchange.+over\b"
    r"|\bgrowth\b|\brank.+against\b|\bbenchmark\b",
    re.I,
)

# ── Time constraint extraction ─────────────────────────────────────────────────

_LAST_N_DAYS_RE = re.compile(r"\blast\s+(\d+)\s+days?\b", re.I)
_LAST_N_MONTHS_RE = re.compile(r"\blast\s+(\d+)\s+months?\b", re.I)
_YEAR_RE = re.compile(r"\b(20\d{2})\b")




# ── Data classes ───────────────────────────────────────────────────────────────

@dataclass
class BusinessIntentPlan:
    """
    Lightweight, stable output of the business intent planner.

    Fields:
      intent      — single action+domain slug ("aggregation", "open_items", …)
      entities    — business nouns extracted from the query, normalized
      behaviors   — active behavioral flags (subset of a fixed vocabulary)
      constraints — concrete values extracted from the query
      confidence  — classifier confidence [0.0, 1.0]
    """
    intent: str
    entities: list[str]
    behaviors: list[str]
    constraints: dict
    confidence: float

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "entities": self.entities,
            "behaviors": self.behaviors,
            "constraints": self.constraints,
            "confidence": self.confidence,
        }


# ── Signal detection ──────────────────────────────────────────────────────────

def _detect_signals(query: str) -> dict:
    top_n_match = _TOP_N_RE.search(query)
    return {
        "has_aggregation": bool(_AGG_RE.search(query)),
        "has_time_filter": bool(_TIME_RE.search(query)),
        "has_open_item": bool(_OPEN_ITEM_RE.search(query)),
        "has_detail_request": bool(_DETAIL_RE.search(query)),
        "has_top_n": bool(top_n_match),
        "top_n_value": int(top_n_match.group(1) or top_n_match.group(2)) if top_n_match else None,
        "is_complex": bool(_COMPLEX_RE.search(query)),
    }


# ── Intent classification ─────────────────────────────────────────────────────

def _classify_intent(signals: dict) -> tuple[str, float]:
    """Return (intent_slug, confidence). Confidence is additive, capped at 0.95."""
    if signals["is_complex"]:
        return "complex_multi_step", 0.50

    conf = 0.30

    if signals["has_open_item"]:
        if signals["has_time_filter"]:
            return "open_items_time_filtered", min(conf + 0.45, 0.95)
        return "open_items", min(conf + 0.30, 0.95)

    if signals["has_top_n"]:
        bonus = 0.25 + (0.15 if signals["has_aggregation"] else 0.0)
        return "top_n_lookup", min(conf + bonus, 0.95)

    if signals["has_aggregation"] and signals["has_time_filter"]:
        return "aggregation_time_filtered", min(conf + 0.40, 0.95)

    if signals["has_aggregation"]:
        return "aggregation", min(conf + 0.30, 0.95)

    if signals["has_detail_request"]:
        return "detail_lookup", min(conf + 0.20, 0.95)

    return "unknown", 0.30


# ── Behavior list ─────────────────────────────────────────────────────────────

def _collect_behaviors(signals: dict) -> list[str]:
    """Map signal flags to a stable behavior vocabulary list."""
    behaviors: list[str] = []
    if signals["has_aggregation"]:
        behaviors.append("aggregation")
    if signals["has_time_filter"]:
        behaviors.append("time_filtered")
    if signals["has_open_item"]:
        behaviors.append("open_items")
    if signals["has_top_n"]:
        behaviors.append("top_n")
    if signals["has_detail_request"] and not signals["has_aggregation"]:
        behaviors.append("detail_rows")
    if signals["is_complex"]:
        behaviors.append("multi_step")
    return behaviors


# ── Constraint extraction ─────────────────────────────────────────────────────

def _extract_constraints(query: str, signals: dict) -> dict:
    """
    Pull concrete constraint values out of the query text.
    Only emits keys that were actually found — no null padding.
    """
    constraints: dict = {}

    if signals["has_top_n"] and signals["top_n_value"] is not None:
        constraints["top_n"] = signals["top_n_value"]

    if signals["has_time_filter"]:
        m_days = _LAST_N_DAYS_RE.search(query)
        m_months = _LAST_N_MONTHS_RE.search(query)
        m_year = _YEAR_RE.search(query)

        if m_days:
            constraints["date_range"] = f"last_{m_days.group(1)}_days"
        elif m_months:
            constraints["date_range"] = f"last_{m_months.group(1)}_months"
        elif re.search(r"\blast\s+month\b", query, re.I):
            constraints["date_range"] = "last_month"
        elif re.search(r"\blast\s+year\b|\bprevious\s+year\b", query, re.I):
            constraints["date_range"] = "last_year"
        elif re.search(r"\bthis\s+month\b|\bcurrent\s+month\b", query, re.I):
            constraints["date_range"] = "this_month"
        elif re.search(r"\bthis\s+year\b|\bcurrent\s+year\b", query, re.I):
            constraints["date_range"] = "this_year"
        elif m_year:
            constraints["date_range"] = m_year.group(1)

    return constraints


# ── Public API ────────────────────────────────────────────────────────────────

async def build_business_intent_plan(query: str) -> BusinessIntentPlan:
    """
    Classify the user query into a structured BusinessIntentPlan.

    Signal detection (behaviors/constraints/intent) is deterministic regex — no LLM cost.
    Entity extraction uses one tiny GPT-4o-mini call for semantic normalization.
    Never raises.

    Args:
        query: Raw user query string.

    Returns:
        BusinessIntentPlan with intent, entities, behaviors, constraints.
    """
    try:
        signals = _detect_signals(query)
        intent, confidence = _classify_intent(signals)
        behaviors = _collect_behaviors(signals)
        constraints = _extract_constraints(query, signals)
        entities = await extract_entities_for_query(query)

        plan = BusinessIntentPlan(
            intent=intent,
            entities=entities,
            behaviors=behaviors,
            constraints=constraints,
            confidence=confidence,
        )

        chat_logger.info(
            "business_intent_planned",
            intent=intent,
            confidence=round(confidence, 2),
            behaviors=behaviors,
            entity_count=len(entities),
            entities=entities[:10],  # cap log output
            constraints=constraints,
        )
        return plan

    except Exception as exc:
        chat_logger.warning("business_intent_plan_error", error=str(exc)[:300])
        return BusinessIntentPlan(
            intent="unknown",
            entities=[],
            behaviors=[],
            constraints={},
            confidence=0.0,
        )
