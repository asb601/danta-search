"""Canonical-master election — PURE decision functions (no DB, no LLM, no I/O).

Per entity-label group, elect the ONE authoritative table for the concept, or
NOTHING. Abstain-biased by construction: a master is written only when the
stored evidence is decisive; every ambiguity leaves the concept blank so the
query layer abstains/asks instead of guessing.

Signals (all data-derived; no hardcoded business vocabulary):
  coverage  — fraction of the member's good_for entries (+ ai_description) that
              mention the concept. good_for is LLM-minted once at ingest
              (UNDERSTAND); scoring it here is deterministic text matching.
  bareness  — the member's FILE NAME tokens reduce to exactly the concept after
              removing the group's own affix tokens (tokens shared by >=
              master_affix_coverage of member names — computed per group, e.g.
              'ap'/'all' on an OEBS dump). Sub-object tables (…_LINES,
              …_DISTRIBUTIONS) keep residue tokens and are never bare.
  rivals    — an eligible member with the OPPOSITE reliable polarity within
              master_twin_margin of the winner forces abstention (AP-vs-AR twin
              ties belong to query-time clarify, never to election).

Decision: eligible = coverage >= master_goodfor_floor; exactly ONE eligible
member may be bare; its score must lead every other eligible member by
master_margin (single-member groups skip the margin). Anything else → None.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass

__all__ = ["MemberEvidence", "ElectionDecision", "elect_master"]

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Score blend. Coverage dominates (it is the authority evidence), bareness is
# the header-vs-subobject separator, key coverage is a weak generic term that
# only matters on real data where the master covers more distinct key values.
_W_COVERAGE = 0.65
_W_BARE = 0.25
_W_KEY = 0.10


@dataclass(frozen=True)
class MemberEvidence:
    file_id: str
    file_name: str                    # File.name (extension ok — tokenizer drops it)
    entity_label: str
    good_for: tuple[str, ...]
    description: str
    row_count: int
    polarity: str | None              # RELIABLE polarity or None (caller gates)
    schema_fingerprint: str | None
    key_cardinality: int              # distinct fingerprints of the entity PK col


@dataclass(frozen=True)
class ElectionDecision:
    label: str
    elected_file_id: str | None
    reason: str                       # "decisive" | abstain reason
    scores: tuple[dict, ...]          # transparency for the dump/eyeball step


def _singular(token: str) -> str:
    """Generic plural→singular normalization (linguistic, not business logic)."""
    if len(token) > 3 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _tokens(text: str) -> set[str]:
    return {_singular(t) for t in _TOKEN_RE.findall((text or "").lower())}


def _name_tokens(file_name: str) -> set[str]:
    stem = file_name.rsplit(".", 1)[0] if "." in file_name else file_name
    return _tokens(stem)


def _coverage(member: MemberEvidence, concept: set[str]) -> float:
    """Fraction of evidence entries that mention the concept. An entry mentions
    the concept when >= half of the concept tokens appear in it (so multi-token
    labels like purchase_order_header still match natural phrasing)."""
    entries = [e for e in member.good_for if e] + ([member.description] if member.description else [])
    if not entries or not concept:
        return 0.0
    need = max(1, (len(concept) + 1) // 2)
    hits = sum(1 for e in entries if len(concept & _tokens(e)) >= need)
    return hits / len(entries)


def _group_affixes(members: list[MemberEvidence], concept: set[str], affix_coverage: float) -> set[str]:
    """Tokens shared by >= ceil(affix_coverage * group size) of member NAMES,
    minus the concept — these are module/template affixes ('ap', 'all'),
    derived from the group's own distribution, never a hardcoded list.

    ceil (not round) so a token in 2 of 4 names is NOT an affix at 0.6 coverage
    (round(2.4)=2 would wrongly make '…_LINES' bare). Multi-member groups need
    at least 2 sharers; a singleton's own non-concept tokens are its affixes
    (that is what makes a single-member label trivially bare).
    """
    if not members:
        return set()
    counts: dict[str, int] = {}
    for m in members:
        for t in _name_tokens(m.file_name):
            counts[t] = counts.get(t, 0) + 1
    floor = max(1 if len(members) == 1 else 2, math.ceil(affix_coverage * len(members)))
    return {t for t, n in counts.items() if n >= floor} - concept


def _is_bare(member: MemberEvidence, concept: set[str], affixes: set[str]) -> bool:
    return not (_name_tokens(member.file_name) - affixes - concept)


def elect_master(label: str, members: list[MemberEvidence], policy) -> ElectionDecision:
    """Pure decision for ONE label group. Returns the elected file_id or None
    with the abstain reason; always returns per-member scores for transparency."""
    concept = _tokens(label)
    affixes = _group_affixes(members, concept, policy.master_affix_coverage)

    max_key = max((m.key_cardinality for m in members), default=0) or 1
    rows: list[dict] = []
    for m in members:
        cov = _coverage(m, concept)
        bare = _is_bare(m, concept, affixes)
        score = (
            _W_COVERAGE * cov
            + _W_BARE * (1.0 if bare else 0.0)
            + _W_KEY * (m.key_cardinality / max_key)
        )
        rows.append({
            "file_id": m.file_id,
            "file_name": m.file_name,
            "score": round(score, 4),
            "coverage": round(cov, 4),
            "bare": bare,
            "eligible": cov >= policy.master_goodfor_floor,
            "polarity": m.polarity,
        })
    scores = tuple(sorted(rows, key=lambda r: r["score"], reverse=True))

    eligible = [r for r in rows if r["eligible"]]
    if not eligible:
        return ElectionDecision(label, None, "no eligible member (good_for not about the concept)", scores)

    bare_eligible = [r for r in eligible if r["bare"]]
    if len(bare_eligible) != 1:
        return ElectionDecision(
            label, None,
            f"{len(bare_eligible)} bare candidates (need exactly 1)", scores,
        )
    winner = bare_eligible[0]

    others = [r for r in eligible if r["file_id"] != winner["file_id"]]
    if others:
        runner_up = max(r["score"] for r in others)
        if winner["score"] - runner_up < policy.master_margin:
            return ElectionDecision(label, None, "margin below master_margin", scores)
        # Opposite-polarity rival close to the winner → AP-vs-AR style tie.
        for r in others:
            if (
                winner["polarity"] and r["polarity"]
                and r["polarity"] != winner["polarity"]
                and winner["score"] - r["score"] < policy.master_twin_margin
            ):
                return ElectionDecision(label, None, "opposite-polarity rival within twin margin", scores)

    return ElectionDecision(label, winner["file_id"], "decisive", scores)
