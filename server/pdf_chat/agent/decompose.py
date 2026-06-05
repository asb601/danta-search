"""Phase-3 — multi-part DECOMPOSITION (agent/decompose.py).

The planner (``planner.py``) sets a ``multi_part`` signal when a query carries
several distinct sub-questions, but that signal was orphaned: ``state.sub_queries``
was never populated, so the loop's ``_components_satisfied`` sufficiency check was
always trivially satisfied and the agent stopped after the first grounded chunk —
silently truncating every multi-part ask. This module closes that gap.

``decompose_query`` splits a multi-part query into its requested OUTPUT COMPONENTS
(the distinct things the answer must cover). It is DATA-DRIVEN: the split is done
by the LLM (the SAME model seam the planner uses — ``model_router.select_model(
task=QUERY_PLANNING)``), never a hardcoded component dictionary or dataset-fitted
hint list. The LLM returns a JSON list of short component labels; on any
backend/parse failure it degrades to a lightweight grounded split (clause
boundaries already present in the query text), so the module never raises and a
multi-part ask is never collapsed back to one implicit component by accident.

Design rules (spec §3 invariants):
  * **No magic literal (4):** the component cap resolves via
    ``get_tunable(container_id, AGENT_DECOMP_MAX_COMPONENTS, ...)`` and the
    decision is emitted via ``log_gate_decision`` (a count compared to a cap is
    never silent).
  * **Per-hop tenant isolation (3):** ``container_id`` scopes the tunable lookup,
    the model-router call, and every gate-decision log.
  * **Honest absence (2):** a failure to decompose degrades to a deterministic
    grounded split rather than guessing or dropping the multi-part intent.

Pure async orchestration over an injected ``llm`` (the ``Llm`` protocol used by
the planner/synthesis). Tests inject an in-memory fake. NEVER raises.
"""
from __future__ import annotations

import json
import re
from typing import Any

import structlog

from ..model_router import TaskClass, select_model
from ..tunables import get_tunable, log_gate_decision

_log = structlog.get_logger("pdf_chat.agent.decompose")

# Named tunable: the maximum number of components a single multi-part ask may be
# split into (bounds fan-out so one prompt can't explode the sufficiency gate).
AGENT_DECOMP_MAX_COMPONENTS = "agent.decomp_max_components"
AGENT_DECOMP_MAX_COMPONENTS_DEFAULT = 6

_DECOMP_SYSTEM = (
    "You split a user's multi-part question into the distinct OUTPUT COMPONENTS "
    "the answer must cover — the separate things being asked. Each component is "
    "a short noun phrase (the subject/metric being requested), NOT a full "
    "sentence. Do not invent components the user did not ask for; do not merge "
    "distinct asks. Reply with ONLY a JSON array of strings, e.g. "
    '["revenue", "headcount", "market share"]. If the question is really a '
    "single ask, return a one-element array."
)

# Clause separators for the deterministic fallback split (structural punctuation
# + the coordinating conjunction "and"). This is grammatical scaffolding, not a
# business/domain dictionary — identical for every tenant.
_FALLBACK_SPLIT_RE = re.compile(r"\s*(?:;|,| and | as well as | plus )\s*", re.IGNORECASE)
_LEADING_QWORDS_RE = re.compile(
    r"^(?:what|which|who|whom|whose|how|when|where|why|is|are|was|were|do|does|"
    r"did|tell me about|show|give|list|summari[sz]e|please|report on|report)\b\s*",
    re.IGNORECASE,
)


def _parse_components(raw: str) -> list[str]:
    """Extract the component JSON array. Tolerates code fences / surrounding prose.

    Raises ``ValueError`` if no JSON array of strings is found — the caller turns
    that into the deterministic fallback split.
    """
    match = re.search(r"\[.*\]", raw, re.DOTALL)
    if not match:
        raise ValueError("no JSON array in decompose reply")
    arr = json.loads(match.group(0))
    if not isinstance(arr, list):
        raise ValueError("decompose JSON is not an array")
    out = [str(x).strip() for x in arr if str(x).strip()]
    if not out:
        raise ValueError("decompose JSON array empty")
    return out


def _clean_component(text: str) -> str:
    """Normalize a fallback clause into a short component label."""
    t = text.strip().strip("?.!").strip()
    t = _LEADING_QWORDS_RE.sub("", t).strip()
    return t


def _fallback_split(query: str) -> list[str]:
    """Deterministic grounded split on clause boundaries already in the query.

    Used only when the LLM split is unavailable/unparseable. Splits on structural
    separators (``;`` ``,`` ``and`` …), strips leading question words, and keeps
    non-empty clauses in order. If nothing splits, returns the single cleaned
    query as one component (so the caller can still treat it as multi-part-aware
    without inventing parts).
    """
    parts = [_clean_component(p) for p in _FALLBACK_SPLIT_RE.split(query or "")]
    parts = [p for p in parts if p]
    if len(parts) >= 2:
        return parts
    single = _clean_component(query or "")
    return [single] if single else []


def _dedupe(components: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for c in components:
        key = c.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    return out


async def decompose_query(
    query: str,
    *,
    container_id: str,
    llm: Any,
    signals: dict | None = None,
) -> list[str]:
    """Split a multi-part ``query`` into its requested output components.

    Returns an ordered, de-duplicated, cap-bounded list of short component
    labels. The split is LLM-driven (model-router seam) with a deterministic
    clause-boundary fallback; both paths route through the same cap. NEVER
    raises — a backend/parse failure degrades to the fallback split.

    A returned list of length ≤ 1 means the query is effectively single-part
    (the caller should NOT populate sub_queries for it — there is nothing to
    gate beyond the implicit whole-query component).
    """
    cap = int(
        get_tunable(
            container_id, AGENT_DECOMP_MAX_COMPONENTS, AGENT_DECOMP_MAX_COMPONENTS_DEFAULT
        )
    )

    # Model selection seam — never hardcode a model id (mirrors the planner).
    sig = dict(signals or {})
    try:
        choice = select_model(
            task=TaskClass.QUERY_PLANNING, container_id=container_id, signals=sig
        )
        sig["planning_model"] = choice.model_id
    except Exception as exc:  # pragma: no cover - router is pure; defensive only
        _log.warning("pdf_chat.decompose.select_model_failed", error=repr(exc))

    components: list[str]
    source: str
    if llm is None:
        components = _fallback_split(query)
        source = "fallback:no_llm"
    else:
        try:
            raw = await llm.generate(
                _DECOMP_SYSTEM, query, container_id=container_id, signals=sig
            )
            components = _parse_components(raw)
            source = "llm"
        except Exception as exc:
            _log.warning("pdf_chat.decompose.llm_split_failed", error=repr(exc))
            components = _fallback_split(query)
            source = f"fallback:{type(exc).__name__}"

    components = _dedupe(components)

    # Cap fan-out (no bare literal: cap came from get_tunable; the decision logs).
    capped = components[:cap]
    log_gate_decision(
        AGENT_DECOMP_MAX_COMPONENTS,
        score=float(len(components)),
        threshold=float(cap),
        outcome="cap" if len(components) > cap else "ok",
        container_id=container_id,
        source=source,
        components=len(capped),
    )
    return capped


__all__ = ["decompose_query", "AGENT_DECOMP_MAX_COMPONENTS"]
