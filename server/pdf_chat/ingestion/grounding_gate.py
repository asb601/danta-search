"""Phase-2 Task 6/11 — the blocking GROUNDING GATE (faithfulness gate).

Spec §1b + §3 invariants 1/5/6: every KG edge AND every tag that the extractor
proposes must be *grounded* in the cited source span before it is allowed to
persist. The gate is the single choke point that enforces this:

    * ``admit_edge`` — a relation is admitted only when its subject, predicate-
      claim, and object are all present (verbatim, normalized) in the cited
      chunk/section span. An ungrounded edge (subject/object/predicate absent
      from the span) is REJECTED (returns ``None``).
    * ``admit_tag`` — a tag is admitted only when its claim text is present in
      the cited span. An ungrounded tag is REJECTED (returns ``None``).
    * ``tag_as_answer`` — the misleading-tag safeguard (spec §1b): a tag is a
      RETRIEVAL signal, never an answer. A tag may only surface as an answer
      claim when it is backed by at least one grounded supporting chunk. With
      no supporting chunk it returns ``None``.

This mirrors the value-overlap + ``edge_provenance`` gate in
``server/app/services/relationship_detector.py`` and the writer-side provenance
contract: no edge/tag without ``src_chunk_id`` + verbatim ``span`` + confidence.

Pure module — zero infra. Inputs are plain attribute-carrying objects
(``ExtractedRelation``/``ExtractedTag`` from ``kg_extraction``, duck-typed so
this module never imports another agent's not-yet-present file). Every gate
decision routes through ``log_gate_decision`` and the tag-confidence floor
resolves through ``get_tunable`` — no bare score-comparison literal lives here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..tunables import get_tunable, log_gate_decision

# Tag confidence floor key (registered in TUNABLE_DEFAULTS by integration; see
# the module return. Passed as a NAMED default here so the gate is self-contained
# and never compares against a bare inline literal — the value flows through
# get_tunable → log_gate_decision exactly like every other pdf_chat threshold).
TUN_TAG_MIN_CONFIDENCE = "kg.tag.min_confidence"
_TAG_MIN_CONFIDENCE_DEFAULT = 0.50

# Word-boundary matching kicks in for SHORT tokens so a 2-char name ("HP", "Q3")
# can't ground against an incidental substring (e.g. "HP" inside "sHiPment"). A
# token at/under this length is matched with \b...\b; longer tokens use plain
# substring containment (whitespace/case normalized). This is the boundary of the
# grounding contract (not a per-container dial), but it flows through
# log_gate_decision so no comparison is silent.
TUN_GROUND_WORD_BOUNDARY_MAX_LEN = "kg.ground.word_boundary_max_len"
_WORD_BOUNDARY_MAX_LEN_DEFAULT = 3

# Structural full-presence gate: a grounded edge requires ALL of its claim
# tokens (subject, object) to be present in the span. Score is the fraction
# present; the admit threshold is full presence. These are invariants of the
# gate's contract (not per-container dials), but they still flow through
# log_gate_decision so no comparison is silent.
_FULL_PRESENCE = 1.0
_ABSENT = 0.0

# A grounded edge starts from a single cited span → one piece of evidence.
_INITIAL_EVIDENCE = 1


# ── grounded artifacts (what the writer persists) ───────────────────────────
@dataclass(frozen=True)
class GroundedEdge:
    """A relation that passed the grounding gate — safe to persist as RELATED_TO.

    Carries the provenance the Neo4j writer binds onto the edge
    (``desc``/``weight``/``confidence``/``evidence_count``/``src_chunk``).
    """

    subject: str
    predicate: str
    obj: str
    confidence: float
    span: str
    src_chunk_id: str
    evidence_count: int


@dataclass(frozen=True)
class GroundedTag:
    """A tag that passed the grounding gate — a RETRIEVAL signal, never an answer.

    ``scope`` is ``"doc"`` or ``"section"``. Surfacing a tag as an answer claim
    requires ``tag_as_answer`` (the misleading-tag safeguard).
    """

    label: str
    scope: str
    confidence: float
    span: str
    src_chunk_id: str


# ── normalization ───────────────────────────────────────────────────────────
_WS = re.compile(r"\s+")


def _norm(text: str) -> str:
    """Lowercase + collapse whitespace so the verbatim-span check is robust to
    incidental spacing/case differences (but not to fabricated claims)."""
    return _WS.sub(" ", (text or "").strip().lower())


def _present(claim: str, haystack_norm: str, *, word_boundary_max_len: int = 0) -> bool:
    """A claim token is grounded when its normalized form appears in the span.

    An empty/whitespace-only claim is NOT grounded (an extractor that emits an
    empty subject/object/label has fabricated the slot — reject it).

    For SHORT tokens (normalized length <= ``word_boundary_max_len``) we require a
    word-boundary match so a 2-char name ("HP"/"Q3") can't ground against an
    incidental substring (e.g. "HP" inside "shipment"). Longer tokens use plain
    substring containment — they are specific enough that an accidental substring
    collision is negligible. ``word_boundary_max_len == 0`` disables the boundary
    rule (plain substring for all lengths).
    """
    needle = _norm(claim)
    if not needle:
        return False
    if word_boundary_max_len and len(needle) <= word_boundary_max_len:
        # \b doesn't anchor next to non-word chars; for tokens whose ends are
        # non-word (rare for names) fall back to substring so we never under-match.
        if needle[0].isalnum() and needle[-1].isalnum():
            return re.search(rf"\b{re.escape(needle)}\b", haystack_norm) is not None
    return needle in haystack_norm


# ── the gate ────────────────────────────────────────────────────────────────
class GroundingGate:
    """Blocking faithfulness gate for KG edges and tags.

    Stateless and pure: construct once and reuse. Every admit/reject decision is
    emitted via ``log_gate_decision`` for auditability.
    """

    def admit_edge(
        self, rel, *, cited_text: str, container_id: str
    ) -> GroundedEdge | None:
        """Admit ``rel`` only if its verbatim PREDICATE ``span`` is present in
        ``cited_text`` AND both endpoints are present.

        Returns a :class:`GroundedEdge` when grounded, else ``None`` (rejected).
        The PREDICATE itself must be verified, not grounded transitively: the
        extractor emits a per-relation ``span`` (the verbatim supporting text for
        THIS relation), so the gate requires that span to be present in the cited
        text. Endpoint co-presence alone is not enough — two names co-occurring in
        a span does not prove the asserted relation between them. An edge whose
        ``span`` (or either endpoint) is absent from the cited text is REJECTED so
        a fabricated predicate can never be persisted.
        """
        cited_norm = _norm(cited_text)
        wb_max = int(
            get_tunable(
                container_id,
                TUN_GROUND_WORD_BOUNDARY_MAX_LEN,
                _WORD_BOUNDARY_MAX_LEN_DEFAULT,
            )
        )
        span_present = _present(
            getattr(rel, "span", ""), cited_norm, word_boundary_max_len=wb_max
        )
        endpoints_present = _present(
            rel.subject, cited_norm, word_boundary_max_len=wb_max
        ) and _present(rel.obj, cited_norm, word_boundary_max_len=wb_max)
        ok = span_present and endpoints_present
        score = _FULL_PRESENCE if ok else _ABSENT
        log_gate_decision(
            "kg.ground.edge",
            score=score,
            threshold=_FULL_PRESENCE,
            outcome="admit" if ok else "reject",
            container_id=container_id,
            subject=rel.subject,
            predicate=rel.predicate,
            obj=rel.obj,
            span_present=span_present,
            endpoints_present=endpoints_present,
            src_chunk_id=rel.src_chunk_id,
        )
        if not ok:
            return None
        return GroundedEdge(
            subject=rel.subject,
            predicate=rel.predicate,
            obj=rel.obj,
            confidence=rel.confidence,
            span=rel.span,
            src_chunk_id=rel.src_chunk_id,
            evidence_count=_INITIAL_EVIDENCE,
        )

    def admit_tag(
        self, tag, *, cited_text: str, container_id: str
    ) -> GroundedTag | None:
        """Admit ``tag`` only if its claim text is present in ``cited_text`` AND
        its confidence clears the per-container tag floor.

        Returns a :class:`GroundedTag` when grounded, else ``None`` (rejected).
        """
        cited_norm = _norm(cited_text)
        grounded = _present(tag.label, cited_norm)
        score = _FULL_PRESENCE if grounded else _ABSENT
        presence = log_gate_decision(
            "kg.ground.tag",
            score=score,
            threshold=_FULL_PRESENCE,
            outcome="admit" if grounded else "reject",
            container_id=container_id,
            label=tag.label,
            scope=tag.scope,
            src_chunk_id=tag.src_chunk_id,
        )
        if not presence["passed"]:
            return None

        floor = get_tunable(
            container_id, TUN_TAG_MIN_CONFIDENCE, _TAG_MIN_CONFIDENCE_DEFAULT
        )
        conf = log_gate_decision(
            "kg.tag.confidence",
            score=tag.confidence,
            threshold=floor,
            outcome="admit" if tag.confidence >= floor else "reject",
            container_id=container_id,
            label=tag.label,
            scope=tag.scope,
        )
        if not conf["passed"]:
            return None

        return GroundedTag(
            label=tag.label,
            scope=tag.scope,
            confidence=tag.confidence,
            span=tag.span,
            src_chunk_id=tag.src_chunk_id,
        )


def _chunk_text(chunk) -> str:
    """Best-effort extraction of a supporting chunk's text.

    A supporting chunk may be a plain string (its text), a dict carrying a
    ``"text"`` key, or an object exposing a ``text`` attribute. Anything else
    contributes no text (so it cannot back a tag).
    """
    if isinstance(chunk, str):
        return chunk
    if isinstance(chunk, dict):
        return str(chunk.get("text", "") or "")
    return str(getattr(chunk, "text", "") or "")


def tag_as_answer(tag, supporting_chunks, *, container_id: str = "") -> str | None:
    """Misleading-tag safeguard (spec §1b): a tag is a retrieval signal, never an
    answer on its own.

    Returns the tag's claim text ONLY when at least one supporting chunk's text
    ACTUALLY CONTAINS the tag label (verified via :func:`_present`) — a non-empty
    supporting list is NOT enough. A tag whose label appears in no supporting
    chunk would be a fabricated answer (the tag asserts something the cited
    evidence never states), so it is suppressed (returns ``None``).

    ``tag`` may be a :class:`GroundedTag` or an extracted tag (duck-typed on
    ``label``). ``supporting_chunks`` is any iterable of grounded supporting
    evidence — each item may be the chunk text (str), a dict with a ``"text"``
    key, or an object exposing a ``text`` attribute.
    """
    label = getattr(tag, "label", None)
    supported = bool(label) and any(
        _present(label, _norm(_chunk_text(chunk))) for chunk in (supporting_chunks or [])
    )
    log_gate_decision(
        "kg.tag.as_answer",
        score=_FULL_PRESENCE if supported else _ABSENT,
        threshold=_FULL_PRESENCE,
        outcome="surface" if supported else "suppress",
        container_id=container_id,
        label=label,
    )
    if not supported:
        return None
    return tag.label
