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

from app.core.llm_tasks import classify_query
from app.core.logger import chat_logger


# ── Behavior signal patterns ───────────────────────────────────────────────────



_TOP_N_RE = re.compile(r"\btop\s+(\d+)\b|\bfirst\s+(\d+)\b", re.I)


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




# ── Constraint extraction ─────────────────────────────────────────────────────

def _extract_constraints(query: str) -> dict:
    """
    Pull concrete constraint values out of the query text.
    Only emits keys that were actually found.
    """

    constraints: dict = {}

    top_match = _TOP_N_RE.search(query)
    if top_match:
        constraints["top_n"] = int(
            top_match.group(1) or top_match.group(2)
        )

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
    try:
        constraints = _extract_constraints(query)

        llm_result = await classify_query(query)

        plan = BusinessIntentPlan(
            intent=llm_result["intent"],
            entities=llm_result["entities"],
            behaviors=llm_result["behaviors"],
            constraints=constraints,
            confidence=llm_result["confidence"],
        )

        chat_logger.info(
            "business_intent_planned",
            intent=plan.intent,
            confidence=round(plan.confidence, 2),
            behaviors=plan.behaviors,
            entity_count=len(plan.entities),
            entities=plan.entities[:10],
            constraints=plan.constraints,
        )

        return plan

    except Exception as exc:
        chat_logger.warning(
            "business_intent_plan_error",
            error=str(exc)[:300],
        )

        return BusinessIntentPlan(
            intent="unknown",
            entities=[],
            behaviors=[],
            constraints={},
            confidence=0.0,
        )