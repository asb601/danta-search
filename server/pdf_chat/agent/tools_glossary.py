"""Phase 5 Task 9 — the ``glossary_lookup`` Tool + transparent ``expand_query``.

This fills the Phase-5 ``glossary_lookup`` SEAM reserved (by name only) in
``pdf_chat/agent/tools.py`` (``RESERVED_TOOL_NAMES``). It does NOT mine the
glossary — it READS the persisted, grounded glossary via the comprehension read
helpers (``pdf_chat/comprehension/reader.py::lookup_glossary``) and maps the row
onto the agent's tool-hit shape. ``register_tool(GlossaryLookupTool())`` runs at
import (the reserved name is accepted; the double-registration guard remains).

Faithfulness contract (spec §4 / invariants 1/2/7):
  * a KNOWN term → expansion + definition + a human provenance LABEL (never a raw
    confidence number) + the grounding citation (chunk_id / bbox);
  * an UNKNOWN term → ONE result with provenance "not found" and NO fabricated
    definition/expansion/citation (refuse, never hallucinate);
  * an ``inferred`` entry surfaces "inferred from usage", NEVER "stated in docs";
  * a ``conflicting`` entry surfaces ALL evidence spans (both sides, no silent
    pick — three-state never resolved at read time).

Tenant isolation: the tool threads ``state.tenant_id`` to the reader on every
lookup — the client never supplies the tenant.

⚠️ WIRING IS DEFERRED. This module registers the tool at import, but activating
it in the live agent requires the agent ``deps`` to carry a ``reader`` (the
comprehension read helpers) and an async ``session``. That wiring belongs in
``agent/graph.py::build_default_deps`` / ``AgentDeps`` — do NOT edit those here.
The Phase-3 ``definitional`` planner branch dispatches this tool by name; its
verbatim-span requirement is satisfied by the glossary ``evidence_spans``.

No score-comparison literal lives in this module: every threshold resolves via
``get_tunable`` and every gate decision is emitted via ``log_gate_decision``
(spec §3 invariant 4). The lookup ``run`` makes NO gate decision (browse, not
gate); ``expand_query`` gates each candidate's stored confidence against
``glossary.min_confidence`` so a low-confidence entry never silently rewrites a
query.
"""
from __future__ import annotations

import re
from typing import Any

from pdf_chat.agent.tools import TOOL_REGISTRY, Tool, register_tool
from pdf_chat.comprehension.provenance import Provenance, label_for
from pdf_chat.tunables import get_tunable, log_gate_decision

# Tunable key (registered in tunables.TUNABLE_DEFAULTS) — the confidence floor a
# glossary entry must clear before expand_query uses it to widen a query.
_TUN_MIN_CONFIDENCE = "glossary.min_confidence"

# Candidate-term tokenizer for expand_query: split the query into word-ish tokens
# (open-vocab — NO jargon allow-list; the glossary itself decides what is a term).
_TERM_RE = re.compile(r"[A-Za-z][A-Za-z0-9.\-_/]*")


# --------------------------------------------------------------------------- #
# deps / row access helpers (rows may be ORM objects, dicts, or namespaces).
# --------------------------------------------------------------------------- #
def _dep(deps: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a deps bundle (attribute or mapping), with a default."""
    if deps is None:
        return default
    val = getattr(deps, key, None)
    if val is None and isinstance(deps, dict):
        val = deps.get(key, default)
    return default if val is None else val


def _field(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict OR an attribute-bearing glossary row."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _citations(row: Any) -> list[dict]:
    """Project a glossary row's ``evidence_spans`` onto citation dicts.

    Each span carries the grounding ``chunk_id`` (+ optional page/bbox/text). A
    CONFLICTING entry keeps ALL spans, so both sides surface here unchanged.
    """
    spans = _field(row, "evidence_spans") or []
    citations: list[dict] = []
    for span in spans:
        citations.append(
            {
                "chunk_id": _field(span, "chunk_id"),
                "page_num": _field(span, "page_num"),
                "bbox": _field(span, "bbox"),
                "text": _field(span, "text"),
            }
        )
    return citations


def _refusal(term: str) -> dict:
    """The single grounded-refusal hit for an unknown term (no fabrication)."""
    return {
        "term": term,
        "expansion": None,
        "definition": None,
        "provenance": label_for(Provenance.NOT_FOUND),
        "citations": [],
    }


def _hit_from_row(term: str, row: Any) -> dict:
    """Map a glossary row → a tool hit (human provenance LABEL, never raw conf)."""
    return {
        "term": _field(row, "term", term),
        "expansion": _field(row, "expansion"),
        "definition": _field(row, "definition"),
        # Human-facing label only — the raw confidence is NEVER surfaced.
        "provenance": label_for(_field(row, "provenance", Provenance.NOT_FOUND.value)),
        "variants": list(_field(row, "variants", []) or []),
        "citations": _citations(row),
    }


# --------------------------------------------------------------------------- #
# The Tool — "what does X mean here?" → expansion + definition + citation.
# --------------------------------------------------------------------------- #
class GlossaryLookupTool:
    """Phase-3 ``Tool``-Protocol adapter for the reserved ``glossary_lookup`` seam.

    ``run(state, deps, *, term)`` reads the persisted glossary via
    ``deps.reader.lookup_glossary(deps.session, state.tenant_id, term)`` and
    returns a ONE-element ``list[dict]`` (the lookup is a single deterministic
    answer, not a ranked hit list). A miss returns a single refusal hit — the
    tool NEVER fabricates a definition. It makes no gate decision (browse, not a
    threshold), so no ``log_gate_decision`` call belongs in ``run``.
    """

    name = "glossary_lookup"

    async def run(self, state, deps, **kw) -> list[dict]:
        term = kw.get("term")
        if term is None and state is not None:
            term = getattr(state, "term", None)
        term = (term or "").strip()
        if not term:
            return [_refusal("")]

        reader = _dep(deps, "reader")
        session = _dep(deps, "session")
        tenant_id = getattr(state, "tenant_id", "") if state is not None else ""
        if reader is None:
            # No reader wired → fail closed with a refusal (never fabricate).
            return [_refusal(term)]

        row = await reader.lookup_glossary(session, tenant_id, term)
        if row is None:
            return [_refusal(term)]
        return [_hit_from_row(term, row)]


# --------------------------------------------------------------------------- #
# Transparent, tenant-scoped query expansion (used by the Phase-3 planner).
# --------------------------------------------------------------------------- #
def _candidate_terms(query: str) -> list[str]:
    """Open-vocab candidate terms from a query (order-preserving, de-duped).

    NO hardcoded jargon list — the glossary itself decides which candidates are
    real terms (a candidate with no glossary row is simply dropped).
    """
    seen: set[str] = set()
    out: list[str] = []
    for tok in _TERM_RE.findall(query or ""):
        if tok not in seen:
            seen.add(tok)
            out.append(tok)
    return out


async def expand_query(
    query: str,
    *,
    tenant_id: str,
    container_id: str,
    reader,
    session=None,
) -> dict:
    """Transparently widen ``query`` with THIS tenant's glossary expansions.

    Returns ``{original, added_terms, expansions:[{term, expansion, provenance}]}``.
    A term is added ONLY when (a) the tenant has a glossary row for it AND (b) the
    row's stored confidence clears ``glossary.min_confidence`` (gated via
    ``log_gate_decision`` so a sub-threshold entry never silently rewrites the
    query). When nothing matches, ``added_terms``/``expansions`` are empty and the
    original is unchanged — the expansion is transparent, never a silent rewrite.

    None-confidence contract (FIX F): a mined row whose stored ``confidence`` is
    ``None`` is treated as already-mined-and-included — it cleared the mining gate
    at ingest, so it is NOT re-gated and DOES expand. That passthrough is made
    TRACEABLE: a ``log_gate_decision`` with ``outcome="no_confidence_passthrough"``
    is emitted (never a silent bypass), so the trace records exactly why a row with
    no confidence still widened the query.

    Tenant-scoped: every lookup carries ``tenant_id`` (the planner trusts the
    token-derived tenant, never client input).
    """
    record = {
        "original": query,
        "added_terms": [],
        "expansions": [],
    }
    if reader is None:
        return record

    threshold = get_tunable(container_id, _TUN_MIN_CONFIDENCE)
    for term in _candidate_terms(query):
        row = await reader.lookup_glossary(session, tenant_id, term)
        if row is None:
            continue
        conf = _field(row, "confidence")
        # Gate the stored confidence against the inclusion floor. A row with no
        # confidence (None) is treated as already-mined-and-included (it cleared
        # the mining gate), so it passes; a present-but-low confidence is logged
        # and dropped from the expansion (no silent rewrite).
        if conf is not None:
            gate = log_gate_decision(
                _TUN_MIN_CONFIDENCE,
                score=conf,
                threshold=threshold,
                outcome="expand_query",
                container_id=container_id,
                tenant_id=tenant_id,
                term=term,
            )
            if not gate["passed"]:
                continue
        else:
            # None-confidence row cleared the mining gate at ingest → it expands,
            # but the passthrough is logged so it is never a SILENT bypass (FIX F).
            # score==threshold ⇒ passed=True, recording the deliberate passthrough.
            log_gate_decision(
                _TUN_MIN_CONFIDENCE,
                score=threshold,
                threshold=threshold,
                outcome="no_confidence_passthrough",
                container_id=container_id,
                tenant_id=tenant_id,
                term=term,
            )
        record["added_terms"].append(term)
        record["expansions"].append(
            {
                "term": _field(row, "term", term),
                "expansion": _field(row, "expansion"),
                "provenance": label_for(
                    _field(row, "provenance", Provenance.NOT_FOUND.value)
                ),
            }
        )
    return record


# --------------------------------------------------------------------------- #
# Registration — fill the reserved seam at import (idempotent on re-import).
# --------------------------------------------------------------------------- #
if GlossaryLookupTool.name not in TOOL_REGISTRY:
    register_tool(GlossaryLookupTool())


__all__ = [
    "GlossaryLookupTool",
    "expand_query",
]
