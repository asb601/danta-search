"""No-LLM backbone for the PDF knowledge graph (Phase-2 Task 3).

Two pure, infra-free proposers that run BEFORE any LLM extraction call so the
expensive ``TaskClass.EXTRACTION`` model only has to *confirm / name / relate*
what the cheap backbone already surfaced (spec §1b):

1. ``propose_entities`` — guarded spaCy NER. spaCy is an optional dependency; the
   import is GUARDED so a tenant/worker without it degrades to an empty list
   (never a crash). An ``nlp`` callable may be injected for testability and to
   reuse a per-process pipeline without re-loading the model per section.

2. ``propose_links`` — value-overlap / co-reference. It reuses the
   ``fingerprint_value`` CONCEPT from ``app/services/relationship_index.py`` —
   normalize a token (trim/lowercase/collapse-space/strip-leading-zeros), drop
   null-like tokens, hash it — and proposes an undirected link between any two
   sections that share a normalized value. No live DB, no global state: the
   fingerprint index is built in-memory from the sections handed in.

Everything here is grounded by construction: each link carries the shared value
as evidence (a verbatim, normalized span the downstream grounding gate can
re-check). Thresholds / caps route through ``get_tunable`` and every prune is
logged via ``log_gate_decision`` (no bare score literal anywhere).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Callable

from pdf_chat.tunables import get_tunable, log_gate_decision

# ── Guarded spaCy import. Absent → backbone degrades to [] (tested both ways).
try:  # pragma: no cover - exercised via monkeypatch in tests
    import spacy  # type: ignore  # noqa: F401

    _HAS_SPACY = True
except Exception:  # ImportError or a broken install must not crash ingestion
    _HAS_SPACY = False

# Lazily-loaded default pipeline, shared per process (loading is expensive).
_DEFAULT_NLP: Any = None

# Tunable keys (registered in tunables.TUNABLE_DEFAULTS by integration — see the
# SHARED-FILE additions in the return). Named defaults are passed inline so this
# module never holds the single source of truth, only a fallback.
TUN_NER_MAX_CANDIDATES = "kg.ner.max_candidates"      # cap entities per section
TUN_LINK_MIN_TOKEN_LEN = "kg.link.min_token_len"      # ignore tiny tokens
TUN_LINK_MAX_VALUE_FANOUT = "kg.link.max_value_fanout"  # drop ubiquitous values

# Null-like tokens that must never become a fingerprint (mirrors the relationship
# index's null handling; kept local so this module needs zero app/DB imports).
_NULL_LIKE = frozenset(
    {"", "n/a", "na", "null", "none", "nan", "nil", "-", "--", "tbd", "unknown"}
)
_LEADING_ZERO_INT_RE = re.compile(r"^0+(\d+)$")
_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-_/.]*")
# A token is "value-like" (a candidate co-reference key) only if it is
# structurally distinctive — contains a digit, an internal id separator, or any
# uppercase letter (proper noun / code). Plain all-lowercase prose words are NOT
# linkable values. This is a STRUCTURAL property of the token, never a hardcoded
# stopword/dictionary list (data-driven per spec).
_VALUE_LIKE_RE = re.compile(r"\d|[-_/.]|[A-Z]")


@dataclass(frozen=True)
class EntityCandidate:
    """A pre-LLM entity proposal. ``label`` is the spaCy ent label (or "")."""

    text: str
    label: str          # spaCy ent label OR ""
    source: str         # "ner" | "value_overlap"


# ──────────────────────────────── helpers ────────────────────────────────────

def _norm_value(value: Any) -> str | None:
    """Normalize a token the same way relationship_index does (concept reuse).

    trim → lowercase → collapse whitespace → strip leading-zero ints → drop
    null-like. Returns None for anything that should not be fingerprinted.
    """
    if value is None:
        return None
    text_value = str(value).strip().lower()
    text_value = re.sub(r"\s+", " ", text_value)
    if text_value in _NULL_LIKE:
        return None
    match = _LEADING_ZERO_INT_RE.match(text_value)
    if match:
        text_value = match.group(1)
    if text_value in _NULL_LIKE:
        return None
    return text_value or None


def fingerprint_value(value: Any) -> str | None:
    """Stable 16-hex fingerprint of a normalized value (None when not keyable)."""
    normalized = _norm_value(value)
    if not normalized:
        return None
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]


def _load_default_nlp() -> Any:
    """Best-effort load of a default spaCy pipeline; None on any failure."""
    global _DEFAULT_NLP
    if _DEFAULT_NLP is not None:
        return _DEFAULT_NLP
    if not _HAS_SPACY:
        return None
    try:  # pragma: no cover - depends on a model being installed
        _DEFAULT_NLP = spacy.load("en_core_web_sm")
    except Exception:
        try:  # blank pipeline still has no NER, but proves the seam degrades
            _DEFAULT_NLP = spacy.blank("en")
        except Exception:
            _DEFAULT_NLP = None
    return _DEFAULT_NLP


# ─────────────────────────────── public API ──────────────────────────────────

def propose_entities(
    text: str, *, container_id: str, nlp: Callable[[str], Any] | None = None
) -> list[EntityCandidate]:
    """Propose entity candidates from ``text`` via guarded spaCy NER.

    Degrades to ``[]`` (no crash) when spaCy is absent AND no ``nlp`` is
    injected. Candidates are case-insensitively de-duplicated and capped at the
    per-container ``kg.ner.max_candidates`` tunable.
    """
    if not text or not text.strip():
        return []

    pipeline = nlp if nlp is not None else (_load_default_nlp() if _HAS_SPACY else None)
    if pipeline is None:
        log_gate_decision(
            "kg.ner.degraded",
            score=0.0,
            threshold=1.0,
            outcome="spacy_absent",
            container_id=container_id,
        )
        return []

    try:
        doc = pipeline(text)
        ents = list(getattr(doc, "ents", []) or [])
    except Exception:  # a broken pipeline must not abort ingestion
        log_gate_decision(
            "kg.ner.degraded",
            score=0.0,
            threshold=1.0,
            outcome="nlp_error",
            container_id=container_id,
        )
        return []

    cap = get_tunable(container_id, TUN_NER_MAX_CANDIDATES, 64)
    seen: set[str] = set()
    out: list[EntityCandidate] = []
    for ent in ents:
        raw = (getattr(ent, "text", "") or "").strip()
        if not raw:
            continue
        key = raw.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(
            EntityCandidate(
                text=raw,
                label=getattr(ent, "label_", "") or "",
                source="ner",
            )
        )
        if len(out) >= cap:
            log_gate_decision(
                "kg.ner.cap",
                score=float(len(out)),
                threshold=float(cap),
                outcome="capped",
                container_id=container_id,
            )
            break
    return out


def propose_links(
    sections: list[Any], *, container_id: str
) -> list[tuple[str, str, str]]:
    """Propose undirected section↔section links via value overlap (co-reference).

    Builds an in-memory fingerprint → {section_id} index from the section text
    (reusing the ``fingerprint_value`` concept), then emits one
    ``(section_a, section_b, evidence)`` per pair of sections sharing a value.
    ``evidence`` is the verbatim normalized shared value (a span the grounding
    gate can re-verify). Ubiquitous values (fan-out above the tunable cap) are
    dropped so a boilerplate token does not link everything to everything.
    """
    if not sections or len(sections) < 2:
        return []

    min_len = get_tunable(container_id, TUN_LINK_MIN_TOKEN_LEN, 3)
    max_fanout = get_tunable(container_id, TUN_LINK_MAX_VALUE_FANOUT, 8)

    # fingerprint -> (normalized_value, ordered list of section_ids)
    index: dict[str, tuple[str, list[str]]] = {}
    for sec in sections:
        sec_id = getattr(sec, "section_id", None)
        sec_text = getattr(sec, "text", "") or ""
        if not sec_id:
            continue
        local_seen: set[str] = set()
        for token in _WORD_RE.findall(sec_text):
            # Only structurally distinctive tokens are co-reference keys; plain
            # lowercase prose words ("are", "red") never link sections.
            if not _VALUE_LIKE_RE.search(token):
                continue
            normalized = _norm_value(token)
            if not normalized or len(normalized) < min_len:
                continue
            fp = fingerprint_value(normalized)
            if not fp or fp in local_seen:
                continue
            local_seen.add(fp)
            value, members = index.setdefault(fp, (normalized, []))
            if sec_id not in members:
                members.append(sec_id)

    links: list[tuple[str, str, str]] = []
    emitted: set[tuple[str, str]] = set()
    for fp, (value, members) in index.items():
        if len(members) < 2:
            continue
        if len(members) > max_fanout:
            log_gate_decision(
                "kg.link.fanout_drop",
                score=float(len(members)),
                threshold=float(max_fanout),
                outcome="drop_ubiquitous",
                container_id=container_id,
                value=value,
            )
            continue
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                a, b = members[i], members[j]
                pair = (a, b) if a <= b else (b, a)
                if pair in emitted:
                    continue
                emitted.add(pair)
                links.append((pair[0], pair[1], value))

    log_gate_decision(
        "kg.link.propose",
        score=float(len(links)),
        threshold=0.0,
        outcome="proposed",
        container_id=container_id,
        sections=len(sections),
    )
    return links
