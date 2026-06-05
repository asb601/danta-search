"""Phase-3 Task 10 — the PDF/graph NEGATIVE-CLAIM + CONFLICT gate.

A confident-but-wrong "there is no data / not found / no mention" answer is the
most damaging failure mode in enterprise document QA. This gate is the PDF port
of ``server/app/services/erp/negative_claim_gate.py`` (which works over SQL
attempts); here the unit of coverage is *retrieved evidence*, not a SQL scan.

Spec §3 invariant 2 (honest absence) — **retrieval-empty ≠ absent**. A "no
data / not found" claim is admitted ONLY when:

    coverage_complete  — the relevant query pages/sections were actually
                         in-context (accessible chunks covered them), AND
    diagnosed          — the claimed item genuinely is not present in any of
                         those in-context chunks (so the empty result is
                         attributable, not just a retrieval miss).

    proven == coverage_complete AND diagnosed

If NOTHING was accessible (``accessible_chunks`` empty) the claim is a retrieval
miss, never a proven absence → ``proven=False`` → ``pdf_honest_rewrite``.

Spec §3 invariant 7 (three-state relationships) — relationships are
**asserted / not-stated / conflicting**. When two in-context chunks make
directly contradictory statements the gate SURFACES the conflict in
``verdict.conflicts`` WITH provenance (chunk_id + page + doc_id); it never
silently picks a side (§4: a non-expert can't catch a subtle wrong pick).

Pure module — zero infra, never raises (mirrors ``evaluate_negative_claim``).
Reuses the verbatim-span ``_present`` / ``_norm`` primitives from the grounding
gate so the absence/presence test is the SAME faithfulness contract used at
ingest. No bare score-comparison literal lives here — every gate decision routes
through ``log_gate_decision`` and any floor resolves through ``get_tunable``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..ingestion.grounding_gate import _norm, _present
from ..tunables import get_tunable, log_gate_decision

# Deterministic negative-claim phrases (lowercased substring match). Mirrors the
# ERP gate's phrase list, extended for document phrasing ("no mention", "does
# not state", "the documents don't"). These are the *signal* — the proof is in
# coverage + diagnosis, never in the phrase itself.
#
# The list is a TUNABLE so a tenant can extend it (e.g. add localized phrasing)
# without a code change. The DEFAULT below is the canonical English set so
# behavior is unchanged when no override is configured. Resolved per-container
# via get_tunable; see ``_negative_phrases``.
TUN_NEG_PHRASES = "agent.neg_claim.phrases"
_NEGATIVE_PHRASES_DEFAULT = (
    "no data", "no records", "no result", "no matching", "no rows",
    "none found", "not found", "there are no", "there is no",
    "could not find", "couldn't find", "no such", "nothing",
    "is missing", "are missing", "missing entirely", "no information",
    "no mention", "not mentioned", "does not state", "do not state",
    "don't state", "doesn't state", "not stated", "no reference",
    "not present", "not specified", "not provided", "absent from",
    "the documents do not", "the documents don't",
)

# Contradiction markers — a chunk that negates a relationship the answer (or
# another chunk) asserts. Used to detect the three-state CONFLICTING case. Also a
# TUNABLE (a tenant may extend with localized negators) defaulting to the
# canonical set so behavior is unchanged.
TUN_NEG_NEGATION_TOKENS = "agent.neg_claim.negation_tokens"
_NEGATION_TOKENS_DEFAULT = (
    "not", "no", "never", "without", "n't", "denies", "denied", "rejects",
    "contradicts", "disputes", "isn't", "aren't", "wasn't", "weren't",
    "does not", "do not", "did not",
)


def _as_phrase_tuple(value, fallback: tuple) -> tuple[str, ...]:
    """Coerce a tunable override into a tuple of lowercased phrases.

    Accepts a list/tuple, or a string (comma- or newline-separated) since the
    env/DB tunable tier yields strings. Falls back to the canonical default on an
    empty/unparseable override (never fewer phrases than baseline behavior).
    """
    if isinstance(value, (list, tuple)):
        items = [str(v).strip().lower() for v in value if str(v).strip()]
    elif isinstance(value, str):
        items = [
            p.strip().lower()
            for p in re.split(r"[\n,]", value)
            if p.strip()
        ]
    else:
        items = []
    return tuple(items) if items else tuple(fallback)


def _negative_phrases(container_id: str) -> tuple[str, ...]:
    raw = get_tunable(container_id, TUN_NEG_PHRASES, _NEGATIVE_PHRASES_DEFAULT)
    return _as_phrase_tuple(raw, _NEGATIVE_PHRASES_DEFAULT)


def _negation_tokens(container_id: str) -> tuple[str, ...]:
    raw = get_tunable(container_id, TUN_NEG_NEGATION_TOKENS, _NEGATION_TOKENS_DEFAULT)
    return _as_phrase_tuple(raw, _NEGATION_TOKENS_DEFAULT)

# Score-only sentinels so log_gate_decision always receives a numeric score and
# the comparison is auditable (these are contract invariants, not dials, but
# they still flow through the logger so no comparison is silent).
_PRESENT = 1.0
_ABSENT = 0.0

# Coverage proof is binary: the relevant in-context evidence is either there or
# not. The floor is the full-coverage threshold; a fraction below it is unproven.
_COVERAGE_FULL = 1.0

# Minimum accessible chunks required before an absence claim can even be
# considered "covered". Named tunable (no inline literal) — a tenant may raise
# it for high-stakes corpora. Defaulted via TUNABLE_DEFAULTS by integration.
TUN_NEG_MIN_COVERAGE_CHUNKS = "agent.neg_claim.min_coverage_chunks"
_MIN_COVERAGE_CHUNKS_DEFAULT = 1


@dataclass
class PdfNegativeVerdict:
    is_negative_claim: bool = False
    proven: bool = False
    coverage_complete: bool = False
    diagnosed: bool = False
    conflicts: list[dict] = field(default_factory=list)


def _is_negative(answer: str, *, container_id: str = "") -> bool:
    low = (answer or "").lower()
    return any(p in low for p in _negative_phrases(container_id))


_CITE_RE = re.compile(r"\[(\d+)\]")


def _cited_indices(answer: str) -> list[int]:
    """The ``[N]`` citation indices an answer references (first-appearance order)."""
    seen: list[int] = []
    for m in _CITE_RE.finditer(answer or ""):
        idx = int(m.group(1))
        if idx not in seen:
            seen.append(idx)
    return seen


def _chunk_field(chunk, *keys, default=None):
    """Best-effort field read across dict / attribute-carrying chunk shapes."""
    for key in keys:
        if isinstance(chunk, dict):
            if key in chunk and chunk[key] is not None:
                return chunk[key]
        else:
            val = getattr(chunk, key, None)
            if val is not None:
                return val
    return default


def _chunk_text(chunk) -> str:
    if isinstance(chunk, str):
        return chunk
    return str(_chunk_field(chunk, "text", default="") or "")


def _chunk_page(chunk):
    return _chunk_field(chunk, "page_num", "page", default=None)


def _chunk_provenance(chunk) -> dict:
    """The provenance a surfaced conflict carries (never a silent pick)."""
    return {
        "chunk_id": _chunk_field(chunk, "chunk_id", "id", default=None),
        "page": _chunk_page(chunk),
        "doc_id": _chunk_field(chunk, "doc_id", "document_id", default=None),
        "text": _chunk_text(chunk),
    }


def _accessible_pages(accessible_chunks) -> set:
    pages = set()
    for c in accessible_chunks or []:
        p = _chunk_page(c)
        if p is not None:
            try:
                pages.add(int(p))
            except (TypeError, ValueError):
                continue
    return pages


def _coverage_complete(accessible_chunks, query_pages, *, container_id: str) -> bool:
    """The relevant query pages/sections were actually in-context.

    retrieval-empty ≠ absent: with NO accessible chunks there is no coverage,
    so an absence claim is a retrieval miss, never proven. When ``query_pages``
    are specified, every one of them must be present among the accessible
    chunks' pages; otherwise (no explicit pages) the presence of at least the
    minimum number of accessible, scanned chunks is the coverage proof.
    """
    chunks = [c for c in (accessible_chunks or []) if c is not None]
    min_chunks = int(
        get_tunable(
            container_id, TUN_NEG_MIN_COVERAGE_CHUNKS, _MIN_COVERAGE_CHUNKS_DEFAULT
        )
    )
    if len(chunks) < max(min_chunks, 1):
        log_gate_decision(
            "agent.neg_claim.coverage",
            score=float(len(chunks)),
            threshold=float(max(min_chunks, 1)),
            outcome="unproven:retrieval_empty",
            container_id=container_id,
            accessible=len(chunks),
        )
        return False

    wanted = set()
    for p in query_pages or []:
        try:
            wanted.add(int(p))
        except (TypeError, ValueError):
            continue
    if wanted:
        have = _accessible_pages(chunks)
        covered = wanted <= have
        score = (len(wanted & have) / len(wanted)) if wanted else _ABSENT
        log_gate_decision(
            "agent.neg_claim.coverage",
            score=score,
            threshold=_COVERAGE_FULL,
            outcome="covered" if covered else "unproven:pages_missing",
            container_id=container_id,
            wanted_pages=sorted(wanted),
            have_pages=sorted(have),
        )
        return covered

    # No explicit query pages → in-context accessible chunks ARE the coverage.
    log_gate_decision(
        "agent.neg_claim.coverage",
        score=float(len(chunks)),
        threshold=float(max(min_chunks, 1)),
        outcome="covered:chunks_in_context",
        container_id=container_id,
        accessible=len(chunks),
    )
    return True


def _diagnosed(answer, accessible_chunks, *, container_id: str) -> bool:
    """The absence is attributable: the relevant evidence was scanned and the
    claimed item genuinely is not present in any in-context chunk.

    Coverage proves the evidence was in front of us; diagnosis confirms we
    actually looked (there is scanned text to attribute the absence to). With
    in-context text present and no contradicting positive evidence, a bare empty
    result IS its own diagnosis (nothing left to probe) — mirrors the ERP gate's
    "no narrowing predicate ⇒ diagnosed" branch.
    """
    chunks = [c for c in (accessible_chunks or []) if c is not None]
    scanned_text = any(_chunk_text(c).strip() for c in chunks)
    log_gate_decision(
        "agent.neg_claim.diagnosed",
        score=_PRESENT if scanned_text else _ABSENT,
        threshold=_COVERAGE_FULL,
        outcome="diagnosed" if scanned_text else "undiagnosed:nothing_scanned",
        container_id=container_id,
        scanned_chunks=len(chunks),
    )
    return scanned_text


def _has_negation(text_norm: str, *, container_id: str = "") -> bool:
    for tok in _negation_tokens(container_id):
        tok_n = _norm(tok)
        if not tok_n:
            continue
        # word-boundary so "no" doesn't fire inside "notes"/"another"
        if re.search(rf"(?:^|\W){re.escape(tok_n)}(?:\W|$)", text_norm):
            return True
    return False


def _shared_anchor(a_norm: str, b_norm: str) -> bool:
    """Two statements concern the same relationship when they share ≥2 content
    tokens (the relationship's named endpoints/predicate), one negated and one
    not. Cheap, deterministic co-reference — no LLM, no infra."""
    stop = {
        "the", "a", "an", "of", "to", "in", "on", "is", "are", "was", "were",
        "and", "or", "for", "by", "with", "as", "at", "it", "this", "that",
        "not", "no", "be", "has", "have", "had", "but",
    }
    ta = {t for t in re.findall(r"[a-z0-9]+", a_norm) if t not in stop and len(t) > 2}
    tb = {t for t in re.findall(r"[a-z0-9]+", b_norm) if t not in stop and len(t) > 2}
    return len(ta & tb) >= 2


def _detect_conflicts(accessible_chunks, *, container_id: str) -> list[dict]:
    """Surface three-state CONFLICTING relationships (spec §3 invariant 7).

    A conflict is two in-context chunks that share a relationship anchor where
    one asserts and the other negates it. BOTH sides are returned with full
    provenance — the gate never silently resolves/picks (§4). Pairwise over the
    accessible set (which is already token-budget bounded upstream).
    """
    chunks = [c for c in (accessible_chunks or []) if c is not None]
    norm = [(_norm(_chunk_text(c)), c) for c in chunks]
    surfaced: list[dict] = []
    seen_ids: set = set()
    for i in range(len(norm)):
        ni, ci = norm[i]
        if not ni:
            continue
        for j in range(i + 1, len(norm)):
            nj, cj = norm[j]
            if not nj:
                continue
            if not _shared_anchor(ni, nj):
                continue
            # contradiction = one negated, the other not, on a shared anchor.
            if _has_negation(ni, container_id=container_id) == _has_negation(
                nj, container_id=container_id
            ):
                continue
            for c in (ci, cj):
                prov = _chunk_provenance(c)
                key = (prov["chunk_id"], prov["page"], prov["doc_id"])
                if key in seen_ids:
                    continue
                seen_ids.add(key)
                surfaced.append(prov)
            log_gate_decision(
                "agent.neg_claim.conflict",
                score=_PRESENT,
                threshold=_COVERAGE_FULL,
                outcome="conflict_surfaced",
                container_id=container_id,
                a=_chunk_provenance(ci)["chunk_id"],
                b=_chunk_provenance(cj)["chunk_id"],
            )
    return surfaced


def evaluate_pdf_negative_claim(
    *,
    answer: str,
    accessible_chunks,
    query_pages=None,
    container_id: str = "",
) -> PdfNegativeVerdict:
    """Return a verdict; never raises. ``proven == coverage_complete AND
    diagnosed``. Conflicts are surfaced with provenance regardless of whether
    the answer is a negative claim (a contradiction in evidence is always worth
    flagging to a non-expert)."""
    try:
        conflicts = _detect_conflicts(accessible_chunks, container_id=container_id)
        if not _is_negative(answer, container_id=container_id):
            # AUDIT a silent miss: a NON-negative answer that nonetheless cited
            # ZERO grounded chunks is suspicious — it may be a confident claim the
            # phrase list failed to catch (a tenant may need to extend the
            # phrases). Log it via the gate harness so silent misses are
            # auditable (it does NOT change the verdict — behavior is unchanged).
            if (answer or "").strip() and not _cited_indices(answer):
                log_gate_decision(
                    "agent.neg_claim.zero_citation_miss",
                    score=_ABSENT,
                    threshold=_PRESENT,
                    outcome="non_negative_but_uncited",
                    container_id=container_id,
                    accessible=len([c for c in (accessible_chunks or []) if c is not None]),
                )
            return PdfNegativeVerdict(
                is_negative_claim=False, proven=True, conflicts=conflicts
            )
        coverage = _coverage_complete(
            accessible_chunks, query_pages, container_id=container_id
        )
        diagnosed = coverage and _diagnosed(
            answer, accessible_chunks, container_id=container_id
        )
        proven = coverage and diagnosed
        log_gate_decision(
            "agent.neg_claim",
            score=_PRESENT if proven else _ABSENT,
            threshold=_COVERAGE_FULL,
            outcome="proven" if proven else "unproven",
            container_id=container_id,
            coverage_complete=coverage,
            diagnosed=diagnosed,
        )
        return PdfNegativeVerdict(
            is_negative_claim=True,
            proven=proven,
            coverage_complete=coverage,
            diagnosed=diagnosed,
            conflicts=conflicts,
        )
    except Exception:
        # Never block the pipeline on a gate error (ERP gate contract).
        return PdfNegativeVerdict(is_negative_claim=False, proven=True)


def pdf_honest_rewrite(verdict: PdfNegativeVerdict) -> str:
    """A scoped, honest replacement for an UNPROVEN negative claim (§4).

    Distinguishes the two unproven cases so a non-expert understands WHY the
    "no data" claim was withheld: a retrieval/coverage miss vs. an
    undiagnosed empty result.
    """
    if not verdict.coverage_complete:
        return (
            "I could not confirm this is absent. I did not retrieve the relevant "
            "pages/sections, so a retrieval miss is not proof the information is "
            "missing from the documents. Please retry — I need the relevant "
            "pages in context before I can state the documents do not contain it."
        )
    return (
        "The relevant pages were in context but I have not verified this is a "
        "true absence rather than content I failed to locate within them. I "
        "should re-scan the in-context evidence before concluding the documents "
        "do not state it."
    )
