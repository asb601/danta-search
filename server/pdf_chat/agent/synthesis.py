"""Phase-3 Task 8/9 — grounded SYNTHESIS with the HARD ENTRY GATE.

This is the single choke point that turns retrieved evidence into a grounded,
citable answer. A confident-but-ungrounded answer is the worst failure mode in
enterprise document QA (spec §0.2), so synthesis enforces a stack of gates:

  1. **Insufficient-context refusal** — with no (or below-floor) accessible
     chunks the synthesizer refuses deterministically and NEVER calls the LLM
     (no hallucination). Mirrors ``graph.llm_generate``.
  2. **Card demotion (spec §1b)** — a section-/doc-CARD hit is a RETRIEVAL
     signal, never quotable evidence. A card contributes ONLY its
     ``src_chunk_ids`` (or its projected source ``chunk_id``) to the context
     pool; the card text is never numbered/cited. The synthesizer quotes only
     real chunk evidence.
  3. **tag_as_answer HARD GATE (spec §1b)** — every tag-/card-derived claim is
     routed through :func:`grounding_gate.tag_as_answer`. A tag whose label
     appears in NO supporting chunk is DROPPED (the gate returns ``None``). Tags
     are a retrieval signal; they may shape what we retrieve, never what we
     assert.
  4. **Citation-density floor (spec §3 inv 1)** — a synthesized answer that
     cites ZERO sources is refused. The floor resolves via
     ``get_tunable("agent.min_citations_per_claim")`` and the decision is logged.
  5. **Provenance labels (spec §4)** — instead of a raw confidence number every
     cited index gets a human-legible label: ``stated`` (a directly grounded
     citation), ``inferred`` (card/tag-derived, not a verbatim quote),
     ``conflicting`` (the negative-claim/conflict gate surfaced a contradiction
     touching it), or ``not_found`` (the model cited an index with no backing
     chunk — surfaced, never silently treated as grounded).
  6. **Staleness hook (spec §4)** — :func:`staleness_annotation` renders a
     "most recent mention is YYYY-MM; may be outdated" note. Hook only: no
     temporal store is wired here.

Pure orchestration over injected ``deps`` (only ``deps.llm`` is used). The LLM
is reached via the ``Llm`` protocol (``PdfLlm.generate`` routes through
``model_router.select_model(task=QUERY_SYNTHESIS)``); tests inject a fake. No
bare score-comparison literal lives here — the citation floor flows through
``get_tunable`` → ``log_gate_decision`` exactly like every other pdf_chat gate.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from pdf_chat.agent.prompts import (
    INSUFFICIENT_CONTEXT_MESSAGE,
    SYSTEM_PROMPT,
    build_user_prompt,
)
from pdf_chat.agent.negative_claim import evaluate_pdf_negative_claim
from pdf_chat.config import get_pdf_settings
from pdf_chat.ingestion.grounding_gate import _norm, _present, tag_as_answer
from pdf_chat.tunables import get_tunable, log_gate_decision

# Citation-density floor: the minimum number of grounded citations a synthesized
# answer must carry. A NAMED default (no inline comparison literal) so the gate
# is import-safe pre-integration; registering it in TUNABLE_DEFAULTS keeps the
# single-source rule (listed as a shared-file addition in the return).
TUN_MIN_CITATIONS = "agent.min_citations_per_claim"
_MIN_CITATIONS_DEFAULT = 1

# Card element types whose hits are a RETRIEVAL signal only — never quotable
# evidence. A card contributes its source chunk ids to the context pool, then is
# dropped. The fixed INTENT-layer set of card kinds (mirrors the searcher's
# section-card / doc-card vector spaces); not a per-tenant dial.
_CARD_ELEMENT_TYPES = frozenset({"section_card", "doc_card", "card"})

# Score sentinels so log_gate_decision always receives a numeric score (these are
# contract invariants, not dials, but they still flow through the logger so no
# comparison is silent).
_PRESENT = 1.0
_ABSENT = 0.0

# Provenance labels (spec §4) — the only legal values.
_STATED = "stated"
_INFERRED = "inferred"
_CONFLICTING = "conflicting"
_NOT_FOUND = "not_found"

_CITE_RE = re.compile(r"\[(\d+)\]")


@dataclass
class SynthesisResult:
    """The grounded synthesis output (plan-locked: answer/citations/provenance).

    ``admitted_tags`` is additive: the tag labels that survived the
    ``tag_as_answer`` HARD GATE (a tag absent from every supporting chunk is
    dropped and never appears here). Exposed so the loop/integration and tests
    can assert the gate actually fired and that an unsupported tag was dropped.
    """

    answer: str
    citations: list[dict] = field(default_factory=list)
    provenance: dict[int, str] = field(default_factory=dict)
    admitted_tags: list[str] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Heterogeneous-chunk helpers (dict or attribute-carrying)
# --------------------------------------------------------------------------- #
def _attr(chunk, name, default=None):
    if isinstance(chunk, dict):
        return chunk.get(name, default)
    return getattr(chunk, name, default)


def _chunk_id(chunk) -> str:
    return str(_attr(chunk, "chunk_id", "") or "")


def _element_type(chunk) -> str:
    et = _attr(chunk, "element_type", "")
    return str(getattr(et, "value", et) or "")


def _is_card(chunk) -> bool:
    return _element_type(chunk) in _CARD_ELEMENT_TYPES


def _src_chunk_ids(chunk) -> list[str]:
    """The chunk ids a card hit demotes to.

    A card carries ``src_chunk_ids`` (its constituent chunks); if absent, the
    searcher projects the card's own source id as ``chunk_id`` (see
    ``neo4j_searcher._card_vector_search``), so that id is the demotion target.
    """
    ids = _attr(chunk, "src_chunk_ids", None)
    if ids:
        return [str(i) for i in ids if i]
    cid = _chunk_id(chunk)
    return [cid] if cid else []


def _is_insufficient(accessible) -> bool:
    """True when too few quotable chunks remain to ground an answer.

    Mirrors ``graph._is_insufficient`` against the configured floor, with a
    local empty-only fallback if the retrieval module is unavailable.
    """
    min_required = get_pdf_settings().min_accessible_chunks
    try:
        from pdf_chat.retrieval.acl import insufficient_context  # type: ignore

        return insufficient_context(accessible, min_required)
    except Exception:
        return len(accessible) < min_required


# --------------------------------------------------------------------------- #
# Context assembly with card demotion + token budget
# --------------------------------------------------------------------------- #
def _quotable_chunks(accessible_chunks) -> list:
    """Resolve the QUOTABLE chunk set: real chunks plus the source chunks a card
    demotes to — but never the card hit itself.

    A card hit pulls its ``src_chunk_ids`` into the context pool. If a source
    chunk is already present as a real accessible chunk it is used as-is (it
    carries text/bbox); otherwise a lightweight placeholder id-only entry is
    added so the source id is reachable for citation but the card text is never
    quoted. Real chunks always win on id (finest-grained provenance).
    """
    real_by_id: dict[str, object] = {}
    ordered: list = []
    for chunk in accessible_chunks or []:
        if chunk is None or _is_card(chunk):
            continue
        cid = _chunk_id(chunk)
        if cid and cid not in real_by_id:
            real_by_id[cid] = chunk
            ordered.append(chunk)

    # Now fold in card demotions — only source ids not already covered by a real
    # chunk (the card text itself is dropped).
    for chunk in accessible_chunks or []:
        if chunk is None or not _is_card(chunk):
            continue
        for src in _src_chunk_ids(chunk):
            if src and src not in real_by_id:
                placeholder = {
                    "chunk_id": src,
                    "text": "",  # card text is NOT quotable evidence
                    "doc_id": _attr(chunk, "doc_id", ""),
                    "page_num": _attr(chunk, "page_num", 0),
                    "bbox": _attr(chunk, "bbox", None),
                    "element_type": "text",
                    "_card_derived": True,
                }
                real_by_id[src] = placeholder
                ordered.append(placeholder)
    return ordered


def _assemble_context(quotable, *, container_id: str) -> tuple[str, list[dict]]:
    """Build the numbered ``[N]`` context block + citation map under a token
    budget (reuses the graph.assemble_context token-guard logic)."""
    budget = get_tunable(container_id, "context_token_budget")
    tokens_per_word = get_tunable(container_id, "context_tokens_per_word")

    def _est_tokens(s: str) -> int:
        return int(len(s.split()) * tokens_per_word)

    lines: list[str] = []
    citations: list[dict] = []
    used_tokens = 0
    n = 0
    for chunk in quotable:
        text = _attr(chunk, "text", "") or ""
        doc_id = _attr(chunk, "doc_id", "")
        page = _attr(chunk, "page_num", 0)
        bbox = _attr(chunk, "bbox", None)
        rendered = f"[{n + 1}] {text}    Source: {doc_id}, page {page}"
        tok = _est_tokens(rendered)
        if n > 0 and used_tokens + tok > budget:
            log_gate_decision(
                "context_token_budget",
                score=used_tokens + tok,
                threshold=budget,
                outcome="truncate",
                container_id=container_id,
                admitted=n,
            )
            break
        n += 1
        used_tokens += tok
        lines.append(f"[{n}] {text}    Source: {doc_id}, page {page}")
        citations.append(
            {
                "n": n,
                "chunk_id": _chunk_id(chunk),
                "doc_id": str(doc_id),
                "page": int(page or 0),
                "bbox": bbox,
                "card_derived": bool(_attr(chunk, "_card_derived", False)),
            }
        )
    return "\n".join(lines), citations


# --------------------------------------------------------------------------- #
# tag_as_answer HARD GATE
# --------------------------------------------------------------------------- #
def _gate_tags(state, quotable, *, container_id: str) -> list[str]:
    """Route every tag-/card-derived claim through ``grounding_gate.tag_as_answer``
    and DROP the unsupported ones (spec §1b HARD ENTRY GATE).

    Supporting evidence is the QUOTABLE chunk set (card hits already demoted to
    their source chunks). A tag whose label appears in NO supporting chunk's text
    is suppressed by the gate (returns ``None``) and never enters the answer. The
    surviving labels are returned so the caller/tests can assert the drop.

    Candidate tags arrive on ``state.candidate_tags`` (an optional, dynamically
    attached signal carried by the retrieval loop); absent ⇒ no tag claims.
    """
    candidate_tags = getattr(state, "candidate_tags", None) or []
    admitted: list[str] = []
    for tag in candidate_tags:
        # `tag_as_answer` is the looked-up symbol (monkeypatchable in tests). It
        # returns the label only when ≥1 supporting chunk actually contains it.
        surfaced = tag_as_answer(tag, quotable, container_id=container_id)
        if surfaced is not None:
            admitted.append(surfaced)
    return admitted


# --------------------------------------------------------------------------- #
# Citation parsing + provenance labelling
# --------------------------------------------------------------------------- #
def _cited_indices(answer: str) -> list[int]:
    """The [N] indices the model actually cited, in first-appearance order."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer or ""):
        idx = int(m.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def _conflicting_pages(state, *, container_id: str) -> set:
    """Pages touched by a SURFACED three-state conflict (never silently
    resolved). Reuses the negative-claim/conflict gate so the conflict signal is
    the SAME contract used elsewhere; the gate never raises.

    The verdict is computed ONCE here and MEMOIZED on ``state.neg_verdict`` so the
    downstream ``negative_claim_node`` reuses it instead of re-running the
    quadratic ``_detect_conflicts`` pass a second time per query (it was being
    computed twice — O(n²) twice). Behavior is identical; only the redundant
    recomputation is removed.
    """
    verdict = evaluate_pdf_negative_claim(
        answer=getattr(state, "answer", "") or "",
        accessible_chunks=getattr(state, "accessible_chunks", []) or [],
        container_id=container_id,
    )
    # Memoize so negative_claim_node does not recompute the conflict pass. Stamp
    # the answer the verdict was computed against so a downstream consumer can
    # tell the memo is still valid (the verdict depends on answer + chunks).
    try:
        setattr(verdict, "_for_answer", getattr(state, "answer", "") or "")
        state.neg_verdict = verdict
    except Exception:  # pragma: no cover - state is a plain dataclass
        pass
    pages: set = set()
    for c in verdict.conflicts or []:
        p = c.get("page")
        if p is not None:
            try:
                pages.add(int(p))
            except (TypeError, ValueError):
                continue
    # Record onto the state so the runtime can surface conflicts downstream.
    try:
        if verdict.conflicts:
            state.conflicts = verdict.conflicts
    except Exception:  # pragma: no cover - state is a plain dataclass
        pass
    return pages


def _label_provenance(
    answer: str,
    citations: list[dict],
    cited_indices: list[int],
    conflict_pages: set,
) -> dict[int, str]:
    """Assign a provenance label per cited index (spec §4).

    * ``not_found``   — a cited [N] with no backing chunk in ``citations``.
    * ``conflicting`` — the backing chunk's page is in a surfaced conflict.
    * ``inferred``    — the backing chunk is card-derived (no verbatim quote).
    * ``stated``      — a directly grounded chunk citation.
    """
    by_n = {c["n"]: c for c in citations}
    provenance: dict[int, str] = {}
    for idx in cited_indices:
        cit = by_n.get(idx)
        if cit is None:
            provenance[idx] = _NOT_FOUND
        elif cit.get("page") in conflict_pages:
            provenance[idx] = _CONFLICTING
        elif cit.get("card_derived"):
            provenance[idx] = _INFERRED
        else:
            provenance[idx] = _STATED
    return provenance


def _refusal(message: str) -> SynthesisResult:
    return SynthesisResult(answer=message, citations=[], provenance={}, admitted_tags=[])


# --------------------------------------------------------------------------- #
# Per-component grounding (multi-part honesty) + staleness
# --------------------------------------------------------------------------- #
def _component_grounding(components, quotable) -> dict[str, bool]:
    """Map each requested output component → whether ≥1 quotable chunk grounds it.

    A component is grounded when its (case-insensitive) label appears in at least
    one quotable chunk's text. This is the SAME containment test the loop's
    ``_components_satisfied`` sufficiency gate uses, so the synthesis-time honesty
    flag is consistent with the loop's stop condition.
    """
    comps = [c for c in (components or []) if c and str(c).strip()]
    haystacks = [(_attr(ch, "text", "") or "").lower() for ch in quotable]
    grounding: dict[str, bool] = {}
    for comp in comps:
        needle = str(comp).strip().lower()
        grounding[str(comp)] = any(needle in h for h in haystacks)
    return grounding


def _partial_components_note(grounding: dict[str, bool]) -> str:
    """A per-component honesty note when a multi-part ask is only partly grounded.

    Lists which requested components ARE grounded and which are NOT, so a partial
    multi-part answer is never emitted as if complete (a silently dropped
    component is the failure this guards against). Empty string when every
    component is grounded (nothing to flag) or there are no components.
    """
    if not grounding:
        return ""
    ungrounded = [c for c, ok in grounding.items() if not ok]
    if not ungrounded:
        return ""
    grounded = [c for c, ok in grounding.items() if ok]
    parts = [
        "Note: this answer does not fully address every part of your question."
    ]
    if grounded:
        parts.append("Grounded in the documents: " + ", ".join(grounded) + ".")
    parts.append(
        "Not found in the retrieved evidence (NOT asserted): "
        + ", ".join(ungrounded)
        + "."
    )
    return " ".join(parts)


def _latest_mention_date(citations, quotable):
    """Best-effort latest mention date from cited chunk metadata (§4 staleness).

    Reads a date-ish field (``date`` / ``published_date`` / ``last_modified`` /
    ``timestamp``) off the quotable chunks backing the emitted citations and
    returns the lexicographically-greatest non-empty value (ISO dates sort
    correctly). Degrades SILENTLY to ``None`` when no chunk carries a date — a
    missing date yields no annotation rather than a misleading one.
    """
    by_id = {_chunk_id(ch): ch for ch in quotable}
    cited_ids = {str(c.get("chunk_id") or "") for c in (citations or [])}
    dates: list[str] = []
    for cid in cited_ids:
        ch = by_id.get(cid)
        if ch is None:
            continue
        for key in ("date", "published_date", "last_modified", "timestamp"):
            val = _attr(ch, key, None)
            if val:
                dates.append(str(val))
                break
    return max(dates) if dates else None


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
async def synthesize(state, deps, *, container_id: str) -> SynthesisResult:
    """Grounded synthesis with the full HARD-ENTRY-GATE stack (spec §1b/§3/§4).

    Returns a :class:`SynthesisResult`. Refuses deterministically (without
    calling the LLM) when there is insufficient accessible context, and refuses
    again — post-generation — when the answer fails the citation-density floor.
    Never raises.
    """
    accessible = getattr(state, "accessible_chunks", []) or []

    # Gate 1 — insufficient context → deterministic refusal, NO LLM call.
    if _is_insufficient(accessible):
        return _refusal(INSUFFICIENT_CONTEXT_MESSAGE)

    # Gate 2 — card demotion: resolve the quotable chunk pool (cards → src ids).
    quotable = _quotable_chunks(accessible)
    if _is_insufficient(quotable):
        return _refusal(INSUFFICIENT_CONTEXT_MESSAGE)

    # Gate 3 — tag_as_answer HARD GATE: drop unsupported tag-/card-derived claims.
    admitted_tags = _gate_tags(state, quotable, container_id=container_id)

    context, citations = _assemble_context(quotable, container_id=container_id)
    state.context = context

    llm = getattr(deps, "llm", None)
    if llm is None:  # no model wired → cannot ground an answer
        return _refusal(INSUFFICIENT_CONTEXT_MESSAGE)

    user = build_user_prompt(state.query, context)
    try:
        answer = await llm.generate(
            SYSTEM_PROMPT,
            user,
            container_id=container_id,
            signals=getattr(state, "router_signals", None) or {},
        )
    except Exception:  # never raise out of synthesis
        return _refusal(INSUFFICIENT_CONTEXT_MESSAGE)

    answer = answer or ""
    state.answer = answer

    cited_indices = _cited_indices(answer)
    # A cited index only "grounds" the answer when it maps to a real citation.
    grounded_cites = [i for i in cited_indices if any(c["n"] == i for c in citations)]

    # Provenance labels (spec §4, NOT raw confidence) are computed for EVERY
    # cited index — including ungrounded ones (labelled not_found) — and surfaced
    # even when the floor refuses, so the caller learns WHY. Conflicts ride the
    # negative-claim/conflict gate so a contradiction is surfaced, never resolved.
    conflict_pages = _conflicting_pages(state, container_id=container_id)
    provenance = _label_provenance(
        answer, citations, cited_indices, conflict_pages
    )
    state.provenance = provenance

    # Gate 4 — citation-density floor: an answer that grounds nothing is refused.
    floor = int(get_tunable(container_id, TUN_MIN_CITATIONS, _MIN_CITATIONS_DEFAULT))
    decision = log_gate_decision(
        "agent.citation_density",
        score=float(len(grounded_cites)),
        threshold=float(floor),
        outcome="admit" if len(grounded_cites) >= floor else "refuse",
        container_id=container_id,
        cited=len(cited_indices),
        grounded=len(grounded_cites),
    )
    if not decision["passed"]:
        # Refuse the CLAIM but still surface the provenance (e.g. the cited
        # indices were not_found) so the absence is honest, not opaque.
        return SynthesisResult(
            answer=(
                "I could not produce a grounded answer: the available evidence "
                "does not support a citable claim, so I am not asserting one."
            ),
            citations=[],
            provenance=provenance,
            admitted_tags=admitted_tags,
        )

    # Keep only the citations the answer actually referenced (grounded set).
    emitted_citations = [c for c in citations if c["n"] in grounded_cites]

    # Multi-part honesty (CRITICAL): if the planner/decomposer requested several
    # output components, flag any that are NOT grounded so a partial answer never
    # silently drops a requested component. The per-component grounding map is
    # recorded on state, and an explicit note is appended to the answer naming
    # which parts are grounded and which are not.
    components = getattr(state, "output_components", None) or getattr(
        state, "sub_queries", None
    ) or []
    grounding = _component_grounding(components, quotable)
    try:
        state.component_grounding = grounding
    except Exception:  # pragma: no cover - state is a plain dataclass
        pass
    partial_note = _partial_components_note(grounding)
    if partial_note:
        log_gate_decision(
            "agent.component_coverage",
            score=float(sum(1 for v in grounding.values() if v)),
            threshold=float(len(grounding)),
            outcome="partial",
            container_id=container_id,
            ungrounded=[c for c, ok in grounding.items() if not ok],
        )
        answer = f"{answer}\n\n{partial_note}"

    # Staleness annotation (§4): a best-effort "may be outdated" note from the
    # latest mention date on the cited chunks' metadata. Degrades silently when
    # no cited chunk carries a date.
    latest = _latest_mention_date(emitted_citations, quotable)
    note = staleness_annotation(latest)
    if note:
        answer = f"{answer}\n\n{note}"

    state.answer = answer
    # Re-stamp the memoized verdict to the FINAL answer so negative_claim_node
    # reuses it (the appended honesty/staleness notes are meta-commentary; the
    # negative-claim verdict was computed against the real grounded claim and is
    # still valid). Avoids a second O(n²) conflict pass.
    memo = getattr(state, "neg_verdict", None)
    if memo is not None:
        try:
            setattr(memo, "_for_answer", answer)
        except Exception:  # pragma: no cover
            pass
    return SynthesisResult(
        answer=answer,
        citations=emitted_citations,
        provenance=provenance,
        admitted_tags=admitted_tags,
    )


# --------------------------------------------------------------------------- #
# Staleness hook (spec §4) — hook only, no temporal store wired
# --------------------------------------------------------------------------- #
def staleness_annotation(latest_date) -> str:
    """Render a "may be outdated" note from the most-recent mention date (§4).

    Hook only: the temporal store is a Phase-5 comprehension artifact, so this
    takes the date as an argument rather than reading it. Returns an empty
    string when no date is known (no annotation rather than a misleading one).
    """
    if not latest_date:
        return ""
    return (
        f"Note: the most recent mention in the documents is {latest_date}; "
        "this information may be outdated."
    )


__all__ = ["SynthesisResult", "synthesize", "staleness_annotation"]
