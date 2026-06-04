"""Phase-2 Task 7 — tests for ENTITY RESOLUTION (per-container bands, unmerged-by-default).

Asserts the conservative resolution contract (spec §1b + §3 inv 1/5):

  * the merge band is DERIVED from the per-container pair-score distribution
    (the ``kg.resolution.merge_band_quantile`` quantile), NOT a hardcoded literal
    — shifting the distribution shifts the band;
  * an ambiguous pair scoring BELOW the per-container band stays UNMERGED
    (unmerged-by-default);
  * a type DISagreement is a hard veto regardless of embedding similarity;
  * there is NO transitive auto-merge below the absolute ``kg.resolution.merge_floor``
    (a weak middle link cannot drag two confidently-distinct entities together);
  * every :class:`MergeDecision` carries its grounding ``evidence``;
  * each :class:`ResolvedEntity` sets ``normalized_value`` (the Phase-4 bridge).

``embed_fn`` is the only infra seam and is MOCKED: a tiny dictionary of unit
vectors so cosine similarity is exact and the band math is deterministic. No
live infra, no spaCy, no Neo4j, no Azure.
"""
from __future__ import annotations

import inspect
import math
import re
from dataclasses import dataclass

import pytest

from pdf_chat.ingestion import entity_resolution as ER
from pdf_chat.ingestion.entity_resolution import (
    EntityResolver,
    MergeDecision,
    ResolvedEntity,
)


# ── plan-exact stand-in input (the ExtractedEntity shape from Task 4) ────────
@dataclass(frozen=True)
class _Entity:
    name: str
    etype: str
    confidence: float = 0.9
    span: str = ""
    src_chunk_id: str = ""


# ── deterministic mock embed_fn ──────────────────────────────────────────────
def _make_embed_fn(vectors: dict[str, list[float]]):
    """Return an embed_fn mapping names → fixed unit-ish vectors (aligned list)."""

    def embed_fn(names: list[str]) -> list[list[float]]:
        return [vectors.get(n, [0.0, 0.0, 0.0]) for n in names]

    return embed_fn


# very-similar pair, a moderate pair, and a far pair — gives a real score spread
_NEAR_A = [1.0, 0.0, 0.0]
_NEAR_B = [0.985, 0.174, 0.0]   # cosine ≈ 0.985 with _NEAR_A
_MID = [0.7, 0.714, 0.0]        # cosine ≈ 0.7 with _NEAR_A
_FAR = [0.0, 0.0, 1.0]          # cosine 0 with _NEAR_A


def test_resolved_entity_and_merge_decision_field_contract():
    """Dataclasses carry exactly the plan's fields (incl. normalized_value/evidence)."""
    re_fields = {f.name for f in ResolvedEntity.__dataclass_fields__.values()}
    assert {"name", "etype", "normalized_value", "aliases"} <= re_fields
    md_fields = {f.name for f in MergeDecision.__dataclass_fields__.values()}
    assert {"kept", "merged", "score", "band", "evidence", "merged_now"} <= md_fields


def test_no_score_literal_in_source():
    """No bare float comparison literal — bands/floors come from get_tunable."""
    src = inspect.getsource(ER)
    # no `score >= 0.8` / `cos < 0.6` style inline thresholds
    assert not re.search(r"(score|cos|sim|band|floor)\s*[<>]=?\s*0\.\d", src)


def test_normalized_value_set_for_phase4_bridge():
    """Every resolved entity exposes a normalized_value (the Phase-4 bridge key)."""
    ents = [_Entity("Acme Corp", "custom:entity_key:org")]
    resolved, _ = EntityResolver().resolve(
        ents, embed_fn=_make_embed_fn({"Acme Corp": _NEAR_A}), container_id="t1"
    )
    assert len(resolved) == 1
    assert resolved[0].normalized_value  # non-empty fingerprint
    assert resolved[0].normalized_value == ER.fingerprint_value("Acme Corp")


def test_band_is_derived_from_score_distribution_not_literal():
    """The same near-duplicate pair MERGES inside a tight cluster but the band
    rises with the distribution — the bar is data-derived, not a constant.

    Cluster A: one very-similar pair among otherwise-far entities → high-quantile
    band sits at the top score → the near pair clears it.
    Cluster B: many near-duplicate pairs → the high-quantile band climbs, so a
    merely-moderate pair that cleared in A would NOT clear here. (Different bands
    for different distributions = derived.)
    """
    container = "t_band"
    # Distribution A: scores ≈ {0.985, 0, 0, ...} → q0.85 lands near the top.
    ents_a = [
        _Entity("Globex", "custom:entity_key:org"),
        _Entity("Globx", "custom:entity_key:org"),
        _Entity("Initech", "custom:entity_key:org"),
        _Entity("Umbrella", "custom:entity_key:org"),
    ]
    embed_a = _make_embed_fn(
        {"Globex": _NEAR_A, "Globx": _NEAR_B, "Initech": _FAR, "Umbrella": _MID}
    )
    _, decisions_a = EntityResolver().resolve(
        ents_a, embed_fn=embed_a, container_id=container
    )
    band_a = decisions_a[0].band

    # Distribution B: every pair is near-identical → q0.85 band is much higher.
    ents_b = [
        _Entity("Globex", "custom:entity_key:org"),
        _Entity("Globx", "custom:entity_key:org"),
        _Entity("Glodex", "custom:entity_key:org"),
    ]
    embed_b = _make_embed_fn(
        {"Globex": _NEAR_A, "Globx": _NEAR_A, "Glodex": _NEAR_A}
    )
    _, decisions_b = EntityResolver().resolve(
        ents_b, embed_fn=embed_b, container_id=container
    )
    band_b = decisions_b[0].band

    # The band tracks the distribution: B (all-identical) has a higher bar than A.
    assert band_b >= band_a
    assert band_a != band_b  # not a fixed literal


def test_ambiguous_pair_below_band_stays_unmerged():
    """A merely-moderate pair below the per-container band is NOT merged."""
    container = "t_amb"
    # One near-duplicate pair (clears) + one moderate pair (ambiguous, below band).
    ents = [
        _Entity("Acme", "custom:entity_key:org"),
        _Entity("Acme Inc", "custom:entity_key:org"),   # near-dup of Acme
        _Entity("Acmoid", "custom:entity_key:org"),      # only moderately similar
    ]
    embed = _make_embed_fn(
        {"Acme": _NEAR_A, "Acme Inc": _NEAR_B, "Acmoid": _MID}
    )
    resolved, decisions = EntityResolver().resolve(
        ents, embed_fn=embed, container_id=container
    )
    # The moderate pair (Acme/Acmoid ≈ 0.7) must be below the q0.85 band → kept.
    moderate = [
        d for d in decisions if {d.kept, d.merged} == {"Acme", "Acmoid"}
    ]
    assert moderate and moderate[0].merged_now is False
    # And the surface form survives as its own canonical (not folded away).
    assert any(r.name == "Acmoid" for r in resolved)


def test_type_disagreement_is_hard_veto():
    """Identical embeddings but different etypes never merge (Apple-co vs Apple-fruit)."""
    ents = [
        _Entity("Apple", "custom:entity_key:org"),
        _Entity("Apple", "custom:entity_key:food"),
    ]
    embed = _make_embed_fn({"Apple": _NEAR_A})  # identical vectors → cosine 1.0
    resolved, decisions = EntityResolver().resolve(
        ents, embed_fn=embed, container_id="t_type"
    )
    assert all(d.merged_now is False for d in decisions)
    assert len(resolved) == 2
    assert all(d.evidence["type_agreement"] is False for d in decisions)


def test_no_transitive_auto_merge_below_floor(monkeypatch):
    """A→B and B→C may each be borderline; if a hop is below the absolute floor
    the chain is broken — no A→C transitive merge through a weak link."""
    container = "t_floor"
    # Force a HIGH floor so only the strongest hop could ever clear it, and a
    # LOW band so the band alone would (wrongly) chain everything together.
    def fake_get_tunable(cid, key, default=None):
        if key == ER.TUN_MERGE_FLOOR:
            return 0.95           # only near-identical pairs clear the floor
        if key == ER.TUN_MERGE_BAND_QUANTILE:
            return 0.0            # band = min score → band alone would chain all
        if key == ER.TUN_COOCCUR_LIFT:
            return 0.0
        return default

    monkeypatch.setattr(ER, "get_tunable", fake_get_tunable)

    # A≈B strongly (clears floor); B≈C only moderately (below floor). Without the
    # floor gate, the low band would union A-B-C into one. With it, C stays out.
    ents = [
        _Entity("NodeA", "custom:entity_key:org"),
        _Entity("NodeB", "custom:entity_key:org"),
        _Entity("NodeC", "custom:entity_key:org"),
    ]
    embed = _make_embed_fn({"NodeA": _NEAR_A, "NodeB": _NEAR_A, "NodeC": _MID})
    resolved, decisions = EntityResolver().resolve(
        ents, embed_fn=embed, container_id=container
    )
    # A and B merge (cosine 1.0 ≥ 0.95 floor); C does NOT join the group.
    bc = [d for d in decisions if {d.kept, d.merged} == {"NodeB", "NodeC"}]
    ac = [d for d in decisions if {d.kept, d.merged} == {"NodeA", "NodeC"}]
    assert bc and bc[0].merged_now is False
    assert ac and ac[0].merged_now is False
    # NodeC survives as its own canonical (not transitively dragged in).
    assert any(r.name == "NodeC" for r in resolved)
    # And A/B did collapse into a single canonical with the other as an alias.
    org_group = [r for r in resolved if r.name in {"NodeA", "NodeB"}]
    assert len(org_group) == 1
    assert "NodeB" in org_group[0].aliases or "NodeA" in org_group[0].aliases


def test_each_merge_decision_carries_evidence():
    """Every MergeDecision exposes per-signal evidence (embedding/type/band/floor)."""
    ents = [
        _Entity("Beta", "custom:entity_key:org"),
        _Entity("Beta LLC", "custom:entity_key:org"),
    ]
    embed = _make_embed_fn({"Beta": _NEAR_A, "Beta LLC": _NEAR_B})
    _, decisions = EntityResolver().resolve(
        ents, embed_fn=embed, container_id="t_ev"
    )
    assert decisions
    for d in decisions:
        assert isinstance(d.evidence, dict)
        assert {"embedding", "type_agreement", "band", "floor"} <= set(d.evidence)


def test_cooccurrence_lifts_a_shared_chunk_pair_over_band():
    """A borderline pair sharing a src_chunk gets a co-occurrence lift that can
    carry it over the band — evidence the two surface forms co-refer."""
    container = "t_cooc"

    def fake_get_tunable(cid, key, default=None):
        if key == ER.TUN_MERGE_BAND_QUANTILE:
            return 0.0          # band = min → band trivially cleared
        if key == ER.TUN_MERGE_FLOOR:
            return 0.72         # _MID cosine ≈0.714 is JUST below; lift pushes over
        if key == ER.TUN_COOCCUR_LIFT:
            return 0.05
        return default

    monkeypatch_ctx = pytest.MonkeyPatch()
    monkeypatch_ctx.setattr(ER, "get_tunable", fake_get_tunable)
    try:
        # same src_chunk_id → co-occurrence lift applies
        ents = [
            _Entity("Xenon", "custom:entity_key:org", src_chunk_id="c1"),
            _Entity("Xenoon", "custom:entity_key:org", src_chunk_id="c1"),
        ]
        embed = _make_embed_fn({"Xenon": _NEAR_A, "Xenoon": _MID})
        _, decisions = EntityResolver().resolve(
            ents, embed_fn=embed, container_id=container
        )
        assert decisions[0].evidence["cooccurrence"] is True
        # base cosine ≈0.714 < 0.72 floor, but +0.05 lift clears it.
        assert decisions[0].merged_now is True
    finally:
        monkeypatch_ctx.undo()


def test_type_disagree_vetoes_do_not_pollute_or_collapse_the_band():
    """The band is derived over the type-AGREEING pair scores only.

    Adding many type-disagreeing (hard-veto, 0-score) pairs must NOT drag the band
    down — otherwise the merge bar collapses on a type-diverse tenant and the
    type-agreeing pairs over-merge. We compare a clean type-agreeing distribution
    against the SAME distribution polluted with veto pairs: the derived band must
    be identical and the merge count must not increase.
    """
    container = "t_pollute"
    embed = _make_embed_fn(
        {
            # type-agreeing org pair: one near pair + one far entity (real spread)
            "Globex": _NEAR_A,
            "Globx": _NEAR_B,
            "Initech": _FAR,
            # extra entities, EACH a UNIQUE type — they form ONLY veto pairs with
            # everyone (no new type-agreeing pair), so they cannot legitimately
            # change the band; if they leak in, it is pure pollution.
            "Mango": _NEAR_A,
            "Banana": _NEAR_A,
        }
    )

    # Clean: only the type-agreeing org entities.
    clean = [
        _Entity("Globex", "custom:entity_key:org"),
        _Entity("Globx", "custom:entity_key:org"),
        _Entity("Initech", "custom:entity_key:org"),
    ]
    resolved_clean, decisions_clean = EntityResolver().resolve(
        clean, embed_fn=embed, container_id=container
    )
    band_clean = decisions_clean[0].band
    merges_clean = sum(1 for d in decisions_clean if d.merged_now)

    # Polluted: same org entities + two entities of UNIQUE distinct types → many
    # type-disagree vetoes, but NO new type-agreeing pair.
    polluted = clean + [
        _Entity("Mango", "custom:entity_key:fruit"),
        _Entity("Banana", "custom:entity_key:legume"),
    ]
    resolved_pol, decisions_pol = EntityResolver().resolve(
        polluted, embed_fn=embed, container_id=container
    )
    band_pol = decisions_pol[0].band
    merges_pol = sum(1 for d in decisions_pol if d.merged_now)

    # The veto pairs are excluded from the band distribution → band unchanged.
    assert band_pol == band_clean
    # And the vetoes never increase the number of merges among the org pairs.
    org_merges_pol = sum(
        1
        for d in decisions_pol
        if d.merged_now and {d.kept, d.merged} <= {"Globex", "Globx", "Initech"}
    )
    assert org_merges_pol == merges_clean
    assert merges_pol == org_merges_pol  # food vetoes contribute zero merges


def test_empty_and_single_inputs_are_safe():
    """Zero entities → empty; one entity → itself with a normalized_value."""
    resolver = EntityResolver()
    assert resolver.resolve([], embed_fn=lambda ns: [], container_id="t") == ([], [])
    resolved, decisions = resolver.resolve(
        [_Entity("Solo", "custom:entity_key:org")],
        embed_fn=_make_embed_fn({"Solo": _NEAR_A}),
        container_id="t",
    )
    assert len(resolved) == 1 and not decisions
    assert resolved[0].normalized_value
