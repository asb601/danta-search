"""Entity resolution for the PDF knowledge graph (Phase-2 Task 7).

Spec §1b + §3 invariants 1/5: collapse the open-vocab entities the extractor
emitted into a smaller set of CANONICAL entities — but conservatively. The
governing principle is **UNMERGED-BY-DEFAULT**: a merge happens only when the
evidence clears a per-container bar that is DERIVED from the data, never a
hardcoded literal. A wrong merge silently fabricates cross-context links across
millions of files and many tenants, so the cost of a false merge is far higher
than the cost of leaving two surface forms separate.

The resolver scores every candidate pair on three grounded signals:

  1. **embedding** similarity (cosine of the per-name embeddings from the
     injected ``embed_fn`` — the SAME model used for ingest/query embeddings).
  2. **type agreement** — both names must share an open-vocab ``etype``
     (``custom:<kind>:<slug>`` mirror of ``app/services/semantic_roles.py``);
     a type DISagreement is a hard veto (an "Apple" company never merges into
     an "Apple" fruit) regardless of embedding similarity.
  3. **co-occurrence** — a small lift when the two surface forms are observed in
     the same source chunk/span (shared ``src_chunk_id``), evidence that they
     co-refer rather than collide by accident.

The merge **band** is the per-container ``kg.resolution.merge_band_quantile``
QUANTILE of the observed pair-score distribution — so the bar adapts to how
separable a given tenant's entities are, and there is NO score literal in the
module. A pair below the band stays unmerged (ambiguous-by-default). Transitive
auto-merge (A→B, B→C ⇒ A→C) is allowed only while every hop also clears the
absolute ``kg.resolution.merge_floor``; below the floor a chain is broken so a
weak link can never drag two confidently-distinct entities together.

Every merge/keep decision is emitted as a :class:`MergeDecision` carrying its
``evidence`` (the per-signal scores + band + floor) and routes through
``log_gate_decision`` — no comparison is silent. ``ResolvedEntity`` sets
``normalized_value`` (via the ``fingerprint_value`` concept) as the Phase-4
bridge key.

Pure module — zero infra. ``embed_fn`` is injected (mock in tests); inputs are
duck-typed on ``name``/``etype``/``src_chunk_id`` so this module never imports
another agent's not-yet-present ``kg_extraction``.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

from pdf_chat.ingestion.ner_backbone import fingerprint_value
from pdf_chat.tunables import get_tunable, log_gate_decision

# ── Tunable keys (registered in TUNABLE_DEFAULTS by integration; see the module
# return). Named defaults are passed inline so this module never holds the single
# source of truth — only a fallback. NO score-comparison literal lives here: the
# band is DERIVED from the per-container score distribution at the quantile below.
TUN_MERGE_BAND_QUANTILE = "kg.resolution.merge_band_quantile"  # band = this quantile of scores
TUN_MERGE_FLOOR = "kg.resolution.merge_floor"                  # absolute floor for (transitive) merge
TUN_COOCCUR_LIFT = "kg.resolution.cooccurrence_lift"           # additive lift for shared-chunk pairs

# Named defaults (mirror these into tunables.TUNABLE_DEFAULTS; the plan lists the
# first two — the co-occurrence lift is this module's tunable).
_DEFAULT_MERGE_BAND_QUANTILE = 0.85
_DEFAULT_MERGE_FLOOR = 0.60
_DEFAULT_COOCCUR_LIFT = 0.05

# A type DISagreement is a hard veto: the pair is never scorable, never merges,
# and is EXCLUDED from the band distribution. Including its 0-score would drag the
# derived band down on a type-diverse tenant (lowering the merge bar → over-merge
# risk on the type-agreeing pairs), so the band quantile is computed over the
# type-agreeing (scorable) pair scores ONLY. These bounds are the contract of a
# cosine-derived score, not per-container dials, but they flow through
# log_gate_decision so nothing is silent.
_SCORE_MIN = 0.0
_SCORE_MAX = 1.0


# ── public artifacts ─────────────────────────────────────────────────────────
@dataclass(frozen=True)
class ResolvedEntity:
    """A canonical entity after resolution.

    ``name``             the kept canonical surface form (the representative).
    ``etype``            the open-vocab ``custom:<kind>:<slug>`` type.
    ``normalized_value`` the Phase-4 bridge key (``fingerprint_value`` of the
                         canonical name); ``None`` when the name is not keyable.
    ``aliases``          the merged-away surface forms (de-duplicated, excludes
                         the canonical name).
    """

    name: str
    etype: str
    normalized_value: str | None
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class MergeDecision:
    """A single keep-vs-merge decision, with its grounding evidence.

    ``kept``       the canonical (representative) name the pair resolves toward.
    ``merged``     the other surface form considered for folding into ``kept``.
    ``score``      the blended pair score (embedding + co-occurrence, type-gated).
    ``band``       the per-container DERIVED merge bar (a score quantile).
    ``evidence``   per-signal breakdown (embedding/type/cooccurrence + floor),
                   so a reviewer can see WHY a pair merged or stayed separate.
    ``merged_now`` whether this decision actually folded ``merged`` into ``kept``.
    """

    kept: str
    merged: str
    score: float
    band: float
    evidence: dict
    merged_now: bool


# ── scoring helpers (pure) ────────────────────────────────────────────────────
def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity clamped to [0, 1] (negative similarity ⇒ not similar)."""
    if not a or not b or len(a) != len(b):
        return _SCORE_MIN
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return _SCORE_MIN
    cos = dot / (na * nb)
    # Clamp: an embedding space may emit slightly-out-of-range values; a negative
    # cosine is "dissimilar", not "anti-similar", for resolution purposes.
    return max(_SCORE_MIN, min(_SCORE_MAX, cos))


def _quantile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated quantile of an already-sorted list (empty ⇒ MAX).

    With no pairs to learn from, returning the maximum makes the band un-clearable
    so the unmerged-by-default posture holds (we never merge on no evidence).
    """
    if not sorted_values:
        return _SCORE_MAX
    if len(sorted_values) == 1:
        return sorted_values[0]
    q = max(0.0, min(1.0, q))
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _entity_name(entity: Any) -> str:
    return (getattr(entity, "name", "") or "").strip()


def _entity_type(entity: Any) -> str:
    return (getattr(entity, "etype", "") or "").strip()


def _entity_chunk(entity: Any) -> str:
    return getattr(entity, "src_chunk_id", "") or ""


class _DSU:
    """Disjoint-set over entity indices for transitive (floor-gated) merging.

    Union is allowed only by the caller AFTER the floor check, so a below-floor
    link never participates in a chain (no transitive auto-merge below the floor).
    """

    def __init__(self, n: int) -> None:
        self._parent = list(range(n))

    def find(self, x: int) -> int:
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        # Keep the lower index as the representative for deterministic canonicals.
        if ra < rb:
            self._parent[rb] = ra
        else:
            self._parent[ra] = rb


# ── the resolver ──────────────────────────────────────────────────────────────
class EntityResolver:
    """Conservative, per-container entity resolver (unmerged-by-default).

    Stateless and pure: construct once and reuse. ``resolve`` is deterministic
    for a given input + ``embed_fn`` (no global state, no infra).
    """

    def resolve(
        self,
        entities: Iterable[Any],
        *,
        embed_fn: Callable[[list[str]], list[Sequence[float]]],
        container_id: str,
    ) -> tuple[list[ResolvedEntity], list[MergeDecision]]:
        """Resolve ``entities`` into canonical :class:`ResolvedEntity` records.

        ``embed_fn`` maps a list of distinct entity names → a list of vectors (one
        per name, aligned by index). It is the ONLY infra seam (mocked in tests).

        Returns ``(resolved, decisions)`` where ``decisions`` records every
        considered pair (merged or kept) with its grounding ``evidence``.
        """
        items = [e for e in entities if _entity_name(e)]
        # De-duplicate identical (name, etype) surface forms up front: two
        # extractions of the exact same name+type are the same entity (not a
        # "merge decision", just a fold). Preserve first-seen order.
        uniq: list[Any] = []
        seen_key: set[tuple[str, str]] = set()
        for e in items:
            key = (_entity_name(e), _entity_type(e))
            if key in seen_key:
                continue
            seen_key.add(key)
            uniq.append(e)

        n = len(uniq)
        if n == 0:
            return [], []
        if n == 1:
            only = uniq[0]
            name = _entity_name(only)
            return (
                [
                    ResolvedEntity(
                        name=name,
                        etype=_entity_type(only),
                        normalized_value=fingerprint_value(name),
                        aliases=[],
                    )
                ],
                [],
            )

        names = [_entity_name(e) for e in uniq]
        types = [_entity_type(e) for e in uniq]
        chunks = [_entity_chunk(e) for e in uniq]

        vectors = list(embed_fn(names))
        cooccur_lift = float(
            get_tunable(container_id, TUN_COOCCUR_LIFT, _DEFAULT_COOCCUR_LIFT)
        )

        # ── score every pair (type disagreement ⇒ hard-veto 0 score) ──────────
        pair_scores: dict[tuple[int, int], float] = {}
        pair_evidence: dict[tuple[int, int], dict] = {}
        # The band is derived over the SCORABLE (type-agreeing) pair scores ONLY.
        # A type-disagreeing pair is a hard veto (score 0, never mergeable), so
        # including its 0 in the distribution would only DRAG THE BAND DOWN on a
        # type-diverse tenant — lowering the merge bar and risking over-merge of
        # the type-agreeing pairs. Veto pairs are excluded from merging anyway, so
        # they must also be excluded from the bar that gates merging.
        scorable_score_dist: list[float] = []
        for i in range(n):
            for j in range(i + 1, n):
                type_agree = bool(types[i]) and types[i] == types[j]
                if not type_agree:
                    emb = _SCORE_MIN
                    score = _SCORE_MIN
                else:
                    vi = vectors[i] if i < len(vectors) else []
                    vj = vectors[j] if j < len(vectors) else []
                    emb = _cosine(vi, vj)
                    cooccur = bool(chunks[i]) and chunks[i] == chunks[j]
                    score = min(_SCORE_MAX, emb + (cooccur_lift if cooccur else 0.0))
                pair_scores[(i, j)] = score
                pair_evidence[(i, j)] = {
                    "embedding": round(emb, 6),
                    "type_agreement": type_agree,
                    "cooccurrence": bool(chunks[i]) and chunks[i] == chunks[j],
                    "etype_a": types[i],
                    "etype_b": types[j],
                }
                if type_agree:
                    scorable_score_dist.append(score)

        # ── DERIVE the band from the per-container SCORABLE score distribution ──
        quantile = float(
            get_tunable(
                container_id, TUN_MERGE_BAND_QUANTILE, _DEFAULT_MERGE_BAND_QUANTILE
            )
        )
        floor = float(get_tunable(container_id, TUN_MERGE_FLOOR, _DEFAULT_MERGE_FLOOR))
        band = _quantile(sorted(scorable_score_dist), quantile)
        log_gate_decision(
            "kg.resolution.band",
            score=band,
            threshold=floor,
            outcome="band_derived",
            container_id=container_id,
            quantile=quantile,
            pair_count=len(scorable_score_dist),
        )

        # ── decide merges: clear the DERIVED band AND the absolute floor ──────
        # The floor (absolute) gates transitivity: a pair below the floor never
        # unions, so a chain can't be dragged together by a weak link. The band
        # (relative) keeps us unmerged-by-default when a tenant's entities are not
        # cleanly separable. A merge requires BOTH bars.
        dsu = _DSU(n)
        decisions: list[MergeDecision] = []
        for (i, j), score in sorted(pair_scores.items()):
            evidence = dict(pair_evidence[(i, j)])
            evidence["band"] = round(band, 6)
            evidence["floor"] = round(floor, 6)
            clears_band = score >= band
            clears_floor = score >= floor
            merged_now = bool(
                pair_evidence[(i, j)]["type_agreement"]
                and clears_band
                and clears_floor
            )
            rec = log_gate_decision(
                "kg.resolution.merge",
                score=score,
                threshold=max(band, floor),
                outcome="merge" if merged_now else "keep_separate",
                container_id=container_id,
                kept=names[i],
                merged=names[j],
                clears_band=clears_band,
                clears_floor=clears_floor,
            )
            evidence["clears_band"] = clears_band
            evidence["clears_floor"] = clears_floor
            evidence["passed"] = rec["passed"]
            decisions.append(
                MergeDecision(
                    kept=names[i],
                    merged=names[j],
                    score=score,
                    band=band,
                    evidence=evidence,
                    merged_now=merged_now,
                )
            )
            if merged_now:
                dsu.union(i, j)

        # ── build canonical entities from the union-find groups ───────────────
        groups: dict[int, list[int]] = {}
        for idx in range(n):
            groups.setdefault(dsu.find(idx), []).append(idx)

        resolved: list[ResolvedEntity] = []
        # Emit in stable order of the canonical (lowest) index.
        for root in sorted(groups):
            members = sorted(groups[root])
            canonical_idx = members[0]
            canonical_name = names[canonical_idx]
            alias_names: list[str] = []
            alias_seen: set[str] = set()
            for m in members:
                if m == canonical_idx:
                    continue
                alias = names[m]
                if alias and alias != canonical_name and alias not in alias_seen:
                    alias_seen.add(alias)
                    alias_names.append(alias)
            resolved.append(
                ResolvedEntity(
                    name=canonical_name,
                    etype=types[canonical_idx],
                    normalized_value=fingerprint_value(canonical_name),
                    aliases=alias_names,
                )
            )

        return resolved, decisions
