"""Phase 5 — the corpus-learned GLOSSARY MINER (Tasks 4/5/6).

``mine_glossary`` turns grounded corpus chunks into ``GlossaryEntry`` rows from
THREE grounded signals, each carrying a faithfulness ``provenance`` + evidence
spans (spec §4; invariants 1/6/7):

  * Task 4 — EXPLICIT definitions → ``STATED``. A regex PROPOSES parenthetical /
    appositive / "stands for" candidates; the proposal is NEVER the decision —
    exactly as ``app/services/column_role_resolver.py`` lets an LLM confirm a
    column's role rather than a heuristic dictionary. The injected LLM CONFIRMS
    each candidate against its supporting span; only a confirmed candidate above
    the ``glossary.min_confidence`` floor becomes a ``STATED`` entry. An
    unconfirmed (or sub-floor) candidate is DROPPED (grounding gate, invariant 1).

  * Task 5 — DISTRIBUTIONAL anomaly → ``INFERRED`` (never ``STATED``). A coined
    term recurs in-corpus FAR ABOVE what general usage predicts. The statistic is
    a homogeneous, VALUE-using "lift": ``lift = corpus_log10freq -
    background_log10freq``, where ``corpus_log10freq = log10(count) -
    log10(total)`` and ``background_log10freq`` is the token's general-usage
    log10-frequency from the INJECTED ``background_freq`` (or a very-rare OOV floor
    ``glossary.background_oov_logfreq`` when absent). A token is a candidate iff
    its lift clears ``glossary.anomaly_min_lift``. Both OOV tokens AND in-table
    tokens are eligible — a frequent general word has a high background log-freq so
    its lift is low and it fails the gate naturally (no function-word
    contamination, no membership pre-filter). The LLM then SYNTHESIZES a usage
    definition. The signal source is INJECTED data (``background_freq`` / the
    shipped ``background_freq.json``) plus corpus stats — NEVER an in-code jargon
    list, so a never-before-seen term is mineable. No background table ⇒ the
    signal is skipped (logged), no crash, no fabricated ``STATED`` entry.

  * Task 6 — CO-REFERENCE variants + CONFLICT. Alias variants of the same term
    collapse into one entry's ``variants[]`` (LLM adjudication, gated by
    ``glossary.coref_min_similarity``). A term with >=2 incompatible confirmed
    expansions becomes ``CONFLICTING``, keeping ALL spans + a recency tag — never
    a silent pick (invariant 7).

Routing: every LLM call goes through ``model_router.select_model(task="synthesis",
signals={})`` — bulk glossary mining is INGESTION BULK, so the data-driven
escalation gate can never fire (``signals={}``) and the strong tier is structurally
unreachable (gpt-4o-mini only; contract C7). An ``assert choice.is_strong is False``
documents + guards the invariant, mirroring ``ingestion/kg_extraction.py``.

Pure-testable with zero infra: the LLM is INJECTED (any object exposing the three
async seam methods); ``background_freq`` is injected (tests pass a dict, prod loads
the shipped JSON). Every threshold resolves via ``get_tunable`` and every gate
decision is emitted via ``log_gate_decision`` — NO score-comparison literal lives
in this module (spec §3 invariant 4).

Call site (deferred wiring): the finalization orchestrator (Phase-1 state machine)
calls ``mine_glossary`` once at ingest finalization, stamped with the new
``ontology_version``; this module does not edit that state machine.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..model_router import TaskClass, select_model
from ..models.comprehension import GlossaryEntry
from ..tunables import get_tunable, log_gate_decision
from .provenance import Provenance

# ── Tunable keys (defaults live in TUNABLE_DEFAULTS — single source) ──────────
TUN_MIN_CONFIDENCE = "glossary.min_confidence"        # inclusion floor
TUN_ANOMALY_MIN_LIFT = "glossary.anomaly_min_lift"    # coined-term lift floor
TUN_BACKGROUND_OOV_LOGFREQ = "glossary.background_oov_logfreq"  # OOV log10-freq
TUN_COREF_MIN_SIMILARITY = "glossary.coref_min_similarity"  # alias-merge floor

# Bulk glossary mining is INGESTION BULK → SYNTHESIS task (router comment:
# "ingestion bulk (community reports, glossary)"). signals={} keeps escalation OFF
# so the strong tier is structurally unreachable; gpt-4o-mini is always returned.
_TASK = TaskClass.SYNTHESIS


def _field(row: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict OR an attribute-bearing row, with a default."""
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default)


def _now() -> datetime:
    return datetime.now(timezone.utc)


# --------------------------------------------------------------------------- #
# Background-frequency table — DATA, loaded from the shipped JSON (open-vocab).
# --------------------------------------------------------------------------- #
_BACKGROUND_PATH = Path(__file__).with_name("background_freq.json")


def load_background_freq() -> dict[str, float]:
    """Load the shipped generic-English/business unigram log-frequency table.

    Returns the table as ``{token: log_freq}`` (the leading ``_comment`` key is
    stripped). Absence/parse-failure ⇒ ``{}`` so the anomaly signal degrades
    gracefully rather than raising. This is the open-vocab signal source: any
    token ABSENT here is a coined-term candidate — no hardcoded jargon list.
    """
    try:
        raw = json.loads(_BACKGROUND_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):  # pragma: no cover - best-effort load
        return {}
    return {k: float(v) for k, v in raw.items()
            if not k.startswith("_") and isinstance(v, (int, float))}


# --------------------------------------------------------------------------- #
# Task 4 — explicit-definition PROPOSER (regex proposes; the LLM decides).
# --------------------------------------------------------------------------- #
# A short-form / initialism token (2+ chars, leading uppercase). Used purely
# STRUCTURALLY to tell the short side from the long side of a paren pair.
_SHORTFORM = re.compile(r"^[A-Z][A-Za-z0-9&./-]{1,}$")
_INITIALISM = re.compile(r"^[A-Z][A-Za-z0-9]+$")

# A long-form WORD allows trailing abbreviation dots ("Cust.", "Acq.").
_LF_WORD = r"[A-Z][A-Za-z]*\.?"
# "Long Form (X)" — words immediately BEFORE a parenthesised token.
_PAREN_AFTER = re.compile(
    rf"({_LF_WORD}(?:\s+{_LF_WORD}){{0,6}})\s*\(([A-Za-z][A-Za-z0-9&./-]+)\)"
)
# "X (Long Form)" — a token immediately BEFORE a parenthesised multiword phrase.
_PAREN_BEFORE = re.compile(
    rf"\b([A-Za-z][A-Za-z0-9&./-]+)\s*\(({_LF_WORD}(?:\s+{_LF_WORD}){{1,6}})\)"
)
# "X stands for Long Form" / "X is short for Long Form".
_STANDS_FOR = re.compile(
    r"\b([A-Z][A-Za-z0-9&./-]+)\b\s+(?:stands for|is short for)\s+"
    r"([A-Z][A-Za-z]+(?:\s+[A-Za-z][A-Za-z]+){0,6})",
    re.IGNORECASE,
)


# A sentence terminator: ``.!?`` + whitespace + an uppercase letter starting a
# NEW word of 2+ lowercase letters. This deliberately does NOT split on
# abbreviation dots ("Cust. Acq. Cost") since the next token there is itself an
# abbreviation, not a lowercase-led sentence start. Structural, open-vocab.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z][a-z])")


def _sentence_of(text: str, needle: str) -> str:
    """Return the sentence in ``text`` containing ``needle`` (else the whole text).

    Splits on real sentence boundaries only (not on abbreviation periods) so a
    definition like "Cust. Acq. Cost (CAC) rose ..." stays intact as one span —
    the verbatim grounding evidence must contain the whole definition.
    """
    for sentence in _SENT_SPLIT.split(text):
        if needle in sentence:
            return sentence.strip()
    return text.strip()


def _align_to_initialism(long_form: str, short: str) -> str:
    """Trim leading words of ``long_form`` so its word-initials spell ``short``.

    Purely STRUCTURAL (open-vocab): when ``short`` is an initialism (say "XYZ")
    and ``long_form`` is "The Xavier Yankee Zulu", the trailing three words'
    initials (X, Y, Z) spell it — so the leading determiner is trimmed to "Xavier
    Yankee Zulu". No stopword/jargon list is consulted; we only align word-initial
    LETTERS to the short token. When no alignment exists the long form is returned
    unchanged (the LLM still confirms downstream).
    """
    if not _INITIALISM.match(short):
        return long_form
    words = long_form.split()
    letters = [c for c in short if c.isalpha()]
    if len(words) <= len(letters):
        return long_form
    # Try the longest trailing window whose initials match the short token.
    tail = words[-len(letters):]
    if [w[0].upper() for w in tail] == [c.upper() for c in letters]:
        return " ".join(tail)
    return long_form


def _propose_explicit(chunk) -> list[dict]:
    """Regex-PROPOSE explicit-definition candidates from one chunk.

    A PROPOSAL only (never the decision): each candidate is ``{term, expansion,
    span, chunk_id, page_num, bbox}``. The LLM confirms downstream. Open-vocab:
    the proposer recognises STRUCTURE (parens / "stands for" / initialism
    alignment), never specific business terms.
    """
    text = _field(chunk, "text", "") or ""
    chunk_id = _field(chunk, "chunk_id", "")
    page_num = _field(chunk, "page_num")
    bbox = _field(chunk, "bbox")
    out: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def _emit(term: str, expansion: str, anchor: str) -> None:
        term, expansion = term.strip(), expansion.strip()
        if not term or not expansion or term.lower() == expansion.lower():
            return
        key = (term, expansion)
        if key in seen:
            return
        seen.add(key)
        out.append({
            "term": term,
            "expansion": _align_to_initialism(expansion, term),
            "span": _sentence_of(text, anchor),
            "chunk_id": chunk_id,
            "page_num": page_num,
            "bbox": bbox,
        })

    # "Long Form (SHORT)" — the parenthesised token is the term.
    for m in _PAREN_AFTER.finditer(text):
        long_form, short = m.group(1), m.group(2)
        if _SHORTFORM.match(short):
            _emit(short, long_form, m.group(0))

    # "SHORT (Long Form)" — the token BEFORE the paren is the term.
    for m in _PAREN_BEFORE.finditer(text):
        short, long_form = m.group(1), m.group(2)
        if _SHORTFORM.match(short):
            _emit(short, long_form, m.group(0))

    # "SHORT stands for Long Form".
    for m in _STANDS_FOR.finditer(text):
        _emit(m.group(1), m.group(2), m.group(0))

    return out


# --------------------------------------------------------------------------- #
# Task 5 — distributional-anomaly candidates (corpus stats vs INJECTED table).
# --------------------------------------------------------------------------- #
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]+")


def _distributional_candidates(
    chunks, background_freq: dict[str, float] | None, container_id: str,
) -> list[dict]:
    """Flag coined terms whose corpus log-frequency LIFTS above general usage.

    The anomaly statistic is a homogeneous, VALUE-using lift::

        corpus_log10freq     = log10(count) - log10(total)
        background_log10freq  = background_freq.get(tok.lower(), oov_floor)
        lift                  = corpus_log10freq - background_log10freq

    where ``oov_floor`` (``glossary.background_oov_logfreq``, very negative) is the
    assumed general-usage log10-frequency of a token ABSENT from the table. A
    coined/jargon term recurs in-corpus far above what general usage predicts ⇒
    HIGH positive lift; a common general word has a high background log-freq that
    cancels its corpus frequency ⇒ low/negative lift, failing the gate naturally
    (no function-word contamination, no membership pre-filter).

    The signal source is INJECTED data (``background_freq``) + corpus statistics
    only — NEVER an in-code dictionary, so a never-before-seen term is mineable
    (open-vocab). Each returned candidate is ``{term, contexts:[{text, chunk_id,
    page_num, bbox}]}``. Absent/empty background ⇒ ``[]`` (signal disabled,
    logged). The lift crossing is decided via ``log_gate_decision`` (no literal).
    """
    if not background_freq:
        log_gate_decision(
            "glossary.anomaly.no_background",
            score=0.0, threshold=1.0, outcome="skip", container_id=container_id,
        )
        return []

    # Corpus-internal term frequency (per distinct surface form, preserving case
    # so coined CamelCase/Proper terms surface as-is).
    counts: dict[str, int] = defaultdict(int)
    contexts: dict[str, list[dict]] = defaultdict(list)
    total = 0
    for ch in chunks:
        text = _field(ch, "text", "") or ""
        seen_here: set[str] = set()
        for tok in _TOKEN.findall(text):
            counts[tok] += 1
            total += 1
            if tok not in seen_here:
                seen_here.add(tok)
                contexts[tok].append({
                    "text": _sentence_of(text, tok),
                    "chunk_id": _field(ch, "chunk_id", ""),
                    "page_num": _field(ch, "page_num"),
                    "bbox": _field(ch, "bbox"),
                })
    if total == 0:
        return []

    min_lift = float(get_tunable(container_id, TUN_ANOMALY_MIN_LIFT))
    oov_floor = float(get_tunable(container_id, TUN_BACKGROUND_OOV_LOGFREQ))
    log_total = math.log10(total)
    out: list[dict] = []
    # EVERY corpus token is eligible (OOV → oov_floor; in-table → its own value).
    # The lift gate filters out general words on the VALUE of their background
    # log-frequency, not on key membership — this is the open-vocab novelty signal.
    for tok, c in counts.items():
        corpus_logfreq = math.log10(c) - log_total
        background_logfreq = background_freq.get(tok.lower(), oov_floor)
        lift = corpus_logfreq - background_logfreq
        decision = log_gate_decision(
            "glossary.anomaly.lift",
            score=lift, threshold=min_lift, outcome="checked",
            container_id=container_id, term=tok,
        )
        if decision["passed"]:
            out.append({"term": tok, "contexts": contexts[tok]})
    return out


# --------------------------------------------------------------------------- #
# Entry construction helpers.
# --------------------------------------------------------------------------- #
def _span_from_candidate(cand: dict, *, expansion: str | None = None) -> dict:
    """Build an evidence-span dict (chunk_id/page_num/bbox/text [+expansion])."""
    span = {
        "chunk_id": cand.get("chunk_id", ""),
        "page_num": cand.get("page_num"),
        "bbox": cand.get("bbox"),
        "text": cand.get("span", ""),
    }
    if expansion is not None:
        span["expansion"] = expansion
    return span


def _entry(
    *, tenant_id, container_id, ontology_version, term, expansion, definition,
    provenance: Provenance, confidence, variants, evidence_spans, first_seen,
) -> GlossaryEntry:
    return GlossaryEntry(
        tenant_id=tenant_id,
        container_id=container_id,
        ontology_version=ontology_version,
        term=term,
        expansion=expansion,
        definition=definition,
        provenance=provenance.value,
        confidence=confidence,
        variants=variants or None,
        evidence_spans=evidence_spans,
        first_seen=first_seen,
    )


# --------------------------------------------------------------------------- #
# Public entry point.
# --------------------------------------------------------------------------- #
async def mine_glossary(
    chunks,
    *,
    llm,
    tenant_id: str,
    container_id: str,
    ontology_version: int = 1,
    background_freq: dict[str, float] | None = None,
) -> list[GlossaryEntry]:
    """Mine grounded ``GlossaryEntry`` rows from ``chunks`` via three signals.

    ``llm`` is the injected seam exposing three async methods:
      * ``confirm_definition(*, term, expansion, span, model_id, container_id)``
        → ``{confirmed: bool, expansion, definition, confidence}``
      * ``synthesize_definition(*, term, contexts, model_id, container_id)``
        → ``{definition, confidence}``
      * ``adjudicate_variants(*, term, candidates, model_id, container_id)``
        → ``{same: bool}``

    Every entry is grounded (carries evidence spans) or never produced (refused).
    Returns unpersisted ORM rows; the finalization orchestrator persists them.
    """
    if not chunks:
        return []

    # Bulk-only routing (escalation OFF by construction); the model id flows to
    # every LLM call. The strong tier is structurally unreachable for SYNTHESIS.
    choice = select_model(task=_TASK, container_id=container_id, signals={})
    assert choice.is_strong is False, "glossary mining must never reach the strong tier"
    model_id = choice.model_id

    min_conf = float(get_tunable(container_id, TUN_MIN_CONFIDENCE))

    # ── Task 4 — propose + LLM-confirm explicit definitions ───────────────────
    # Group confirmed candidates by term so Task 6 can reconcile variants/conflict.
    confirmed_by_term: dict[str, list[dict]] = defaultdict(list)
    for ch in chunks:
        for cand in _propose_explicit(ch):
            verdict = await llm.confirm_definition(
                term=cand["term"], expansion=cand["expansion"], span=cand["span"],
                model_id=model_id, container_id=container_id,
            ) or {}
            if not verdict.get("confirmed"):
                log_gate_decision(
                    "glossary.explicit.unconfirmed",
                    score=0.0, threshold=1.0, outcome="drop",
                    container_id=container_id, term=cand["term"],
                )
                continue
            conf = float(verdict.get("confidence", 0.0))
            decision = log_gate_decision(
                TUN_MIN_CONFIDENCE,
                score=conf, threshold=min_conf, outcome="checked",
                container_id=container_id, term=cand["term"],
            )
            if not decision["passed"]:
                continue
            confirmed_by_term[cand["term"]].append({
                **cand,
                "expansion": verdict.get("expansion", cand["expansion"]),
                "definition": verdict.get("definition"),
                "confidence": conf,
                "doc_date": _field(ch, "doc_date"),
            })

    entries: list[GlossaryEntry] = []
    for term, cands in confirmed_by_term.items():
        entries.append(
            await _reconcile_term(
                term, cands, llm=llm, model_id=model_id,
                tenant_id=tenant_id, container_id=container_id,
                ontology_version=ontology_version,
            )
        )

    explicit_terms = set(confirmed_by_term)

    # ── Task 5 — distributional anomaly → INFERRED (never STATED) ──────────────
    for cand in _distributional_candidates(chunks, background_freq, container_id):
        term = cand["term"]
        if term in explicit_terms:
            continue  # an explicit STATED definition already won this term
        contexts = cand["contexts"]
        result = await llm.synthesize_definition(
            term=term, contexts=[c["text"] for c in contexts],
            model_id=model_id, container_id=container_id,
        ) or {}
        definition = result.get("definition")
        conf = float(result.get("confidence", 0.0))
        decision = log_gate_decision(
            TUN_MIN_CONFIDENCE,
            score=conf, threshold=min_conf, outcome="checked",
            container_id=container_id, term=term, signal="distributional",
        )
        if not (definition and decision["passed"]):
            continue
        evidence_spans = [
            {"chunk_id": c["chunk_id"], "page_num": c["page_num"],
             "bbox": c["bbox"], "text": c["text"]}
            for c in contexts
        ]
        entries.append(_entry(
            tenant_id=tenant_id, container_id=container_id,
            ontology_version=ontology_version, term=term, expansion=None,
            definition=definition, provenance=Provenance.INFERRED,
            confidence=conf, variants=None, evidence_spans=evidence_spans,
            first_seen=_now(),
        ))

    return entries


# --------------------------------------------------------------------------- #
# Task 6 — co-reference variants + conflict reconciliation for ONE term.
# --------------------------------------------------------------------------- #
async def _reconcile_term(
    term: str, cands: list[dict], *, llm, model_id: str,
    tenant_id: str, container_id: str, ontology_version: int,
) -> GlossaryEntry:
    """Reconcile all confirmed candidates for ``term`` into a single entry.

    A single (or consistent) expansion ⇒ ``STATED`` with the alias surface forms
    collapsed into ``variants[]``. Two or more INCOMPATIBLE expansions ⇒
    ``CONFLICTING`` keeping ALL spans + a recency tag (never a silent pick).
    """
    distinct_expansions = list(dict.fromkeys(c["expansion"] for c in cands if c["expansion"]))
    first_seen = _now()

    # Single candidate / single expansion → STATED (no adjudication needed).
    if len(distinct_expansions) <= 1:
        return _stated_entry(
            term, cands, tenant_id=tenant_id, container_id=container_id,
            ontology_version=ontology_version, first_seen=first_seen,
        )

    # >=2 distinct expansions: adjudicate whether they corefer (same meaning,
    # alias spelling) or conflict (incompatible meanings). The similarity floor
    # is logged; the LLM's verdict is the adjudication.
    sim_floor = float(get_tunable(container_id, TUN_COREF_MIN_SIMILARITY))
    log_gate_decision(
        TUN_COREF_MIN_SIMILARITY,
        score=1.0, threshold=sim_floor, outcome="adjudicate",
        container_id=container_id, term=term, n_expansions=len(distinct_expansions),
    )
    verdict = await llm.adjudicate_variants(
        term=term, candidates=distinct_expansions,
        model_id=model_id, container_id=container_id,
    ) or {}

    if verdict.get("same"):
        return _stated_entry(
            term, cands, tenant_id=tenant_id, container_id=container_id,
            ontology_version=ontology_version, first_seen=first_seen,
        )

    # CONFLICTING — keep ALL spans (both sides), recency-tagged by doc date.
    log_gate_decision(
        "glossary.conflict",
        score=float(len(distinct_expansions)), threshold=1.0, outcome="conflict",
        container_id=container_id, term=term,
    )
    evidence_spans = []
    for c in cands:
        span = _span_from_candidate(c, expansion=c["expansion"])
        span["definition"] = c.get("definition")
        span["doc_date"] = c.get("doc_date")
        evidence_spans.append(span)
    # Recency: the newest dated candidate's expansion is tagged (surfaced, not
    # silently chosen — all sides remain in evidence_spans).
    most_recent = max(
        (c for c in cands if c.get("doc_date")),
        key=lambda c: str(c["doc_date"]), default=cands[-1],
    )
    confidence = max(c["confidence"] for c in cands)
    return _entry(
        tenant_id=tenant_id, container_id=container_id,
        ontology_version=ontology_version, term=term,
        expansion=most_recent.get("expansion"),
        definition=most_recent.get("definition"), provenance=Provenance.CONFLICTING,
        confidence=confidence, variants=distinct_expansions,
        evidence_spans=evidence_spans, first_seen=first_seen,
    )


def _stated_entry(
    term: str, cands: list[dict], *, tenant_id, container_id, ontology_version,
    first_seen,
) -> GlossaryEntry:
    """Build a STATED entry, collapsing alias surface forms into ``variants[]``."""
    primary = max(cands, key=lambda c: c["confidence"])
    expansion = primary["expansion"]
    # Variants = alias surface forms seen for this term across chunks (the spans'
    # leading text differs from the canonical expansion).
    variants: list[str] = []
    for c in cands:
        alias = _alias_of(c["span"], term)
        if alias and alias not in variants and alias != expansion:
            variants.append(alias)
    evidence_spans = [_span_from_candidate(c, expansion=c["expansion"]) for c in cands]
    confidence = max(c["confidence"] for c in cands)
    return _entry(
        tenant_id=tenant_id, container_id=container_id,
        ontology_version=ontology_version, term=term, expansion=expansion,
        definition=primary.get("definition"), provenance=Provenance.STATED,
        confidence=confidence, variants=variants, evidence_spans=evidence_spans,
        first_seen=first_seen,
    )


def _alias_of(span: str, term: str) -> str | None:
    """Extract the long-form surface preceding ``(term)`` in ``span`` as an alias.

    Open-vocab: this reads the STRUCTURE of the span (the words before the
    parenthesised term), never a known alias table. Returns ``None`` when the
    span has no parenthetical long form.
    """
    pat = re.compile(
        r"([A-Za-z][A-Za-z.&/ -]*?)\s*\(\s*" + re.escape(term) + r"\s*\)"
    )
    m = pat.search(span or "")
    if not m:
        return None
    return m.group(1).strip(" .") or None


__all__ = [
    "mine_glossary",
    "load_background_freq",
    "_distributional_candidates",
]
