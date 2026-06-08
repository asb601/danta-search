"""Value-corroboration guard on the entity-resolver's HARD pin (pure, no DB).

Today the resolver hard-pins a table at confidence >= resolver_pin_threshold on
NAME / ROLE-LABEL token overlap with ZERO value check, so a similarly-named-but-
wrong table (OE_SO_HOLDS_ALL over OE_ORDER_HEADERS_ALL; OTA_VENDOR_SUPPLIES for a
vendor query) gets prepended to the shortlist and can displace the correct table
— which also silently kills the P1 joins (both join endpoints must be in the
shortlist).

The guard is NOT a re-scorer. It is a GATE on the existing pin promotion: a
candidate may be HARD-pinned only if its match is CORROBORATED BY VALUE EVIDENCE
— the column the resolver matched the entity to (or a column on that table
carrying the entity's key role) must be a genuine key per `column_key_registry`:
  cardinality >= policy.min_join_cardinality (8)  AND  key_kind in {pk, fk}.

Rules under test:
  1. Corroborated candidate (key column cardinality 200, key_kind fk) at conf
     0.95 -> hard-pinned (kept).
  2. Name-only candidate at conf 0.95 whose matched column has cardinality 3 /
     not a key (or absent in registry) -> NOT hard-pinned (demoted).
  3. Two candidates equally corroborated (both key cardinality 312, fk) ->
     neither force-pinned (abstain — honest clone behavior, no name heuristic).
  4. Flag OFF -> legacy behavior (pin purely on confidence >= threshold).

Run: cd server && uv run python -m pytest testing/test_resolver_pin_value_guard.py -q
"""
from __future__ import annotations

from app.services.entity_resolver import (
    EntityCandidate,
    RegistryKeyEvidence,
    apply_pin_value_guard,
    select_pinnable_blobs,
)

PIN_THRESHOLD = 0.85
MIN_CARD = 8  # SemanticPolicy.min_join_cardinality


def _cand(table: str, conf: float, *, corroborated: bool | None = None) -> EntityCandidate:
    return EntityCandidate(
        table=table,
        file_id=table,  # 1:1 for the fixture
        confidence=conf,
        reason="semantic_role_match",
        corroborated=corroborated,
    )


# ── Corroboration check (registry evidence -> bool) ─────────────────────────────

def test_genuine_fk_key_corroborates():
    # A real key: high cardinality, key_kind fk, role label matches the entity.
    rows = [
        RegistryKeyEvidence(
            column_name="VENDOR_ID",
            semantic_role="custom:entity_key:vendor",
            key_kind="fk",
            cardinality=200,
        )
    ]
    assert apply_pin_value_guard(("vendor",), rows, MIN_CARD) is True


def test_low_cardinality_enum_does_not_corroborate():
    # Name/role matches but the column is a 3-value enum -> not a real key.
    rows = [
        RegistryKeyEvidence(
            column_name="VENDOR_STATUS",
            semantic_role="custom:entity_key:vendor",
            key_kind="fk",
            cardinality=3,
        )
    ]
    assert apply_pin_value_guard(("vendor",), rows, MIN_CARD) is False


def test_absent_from_registry_does_not_corroborate():
    # No registry rows for the file at all -> no value evidence.
    assert apply_pin_value_guard(("vendor",), [], MIN_CARD) is False


def test_key_for_a_different_entity_does_not_corroborate():
    # A genuine key, but it carries a DIFFERENT entity's role -> not evidence
    # for THIS entity. (No name heuristic; pure role-label overlap.)
    rows = [
        RegistryKeyEvidence(
            column_name="CUSTOMER_ID",
            semantic_role="custom:entity_key:customer",
            key_kind="fk",
            cardinality=200,
        )
    ]
    assert apply_pin_value_guard(("vendor",), rows, MIN_CARD) is False


# ── Pin selection (the GATE on the existing pin promotion) ──────────────────────

def test_corroborated_candidate_is_pinned():
    resolution = {
        "vendor": [
            _cand("OE_ORDER_HEADERS_ALL.parquet", 0.95, corroborated=True),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=True)
    assert pinned == {"OE_ORDER_HEADERS_ALL.parquet"}


def test_name_only_candidate_is_demoted():
    # conf 0.95 (>= threshold) but value evidence is missing -> NOT pinned.
    resolution = {
        "vendor": [
            _cand("OTA_VENDOR_SUPPLIES.parquet", 0.95, corroborated=False),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=True)
    assert pinned == set()


def test_equally_corroborated_clones_abstain():
    # Two byte-identical clones, both genuine fk cardinality 312 -> abstain.
    resolution = {
        "vendor": [
            _cand("AP_BATCHES_ALL.parquet", 0.95, corroborated=True),
            _cand("AP_BATCHES_ALL_COPY.parquet", 0.95, corroborated=True),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=True)
    assert pinned == set()


def test_corroborated_wins_over_uncorroborated_clone():
    # If only one of the two at-threshold candidates is corroborated, pin it.
    resolution = {
        "vendor": [
            _cand("OE_SO_HOLDS_ALL.parquet", 0.95, corroborated=False),
            _cand("OE_ORDER_HEADERS_ALL.parquet", 0.95, corroborated=True),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=True)
    assert pinned == {"OE_ORDER_HEADERS_ALL.parquet"}


def test_below_threshold_never_pinned_even_if_corroborated():
    resolution = {
        "vendor": [
            _cand("OE_ORDER_HEADERS_ALL.parquet", 0.50, corroborated=True),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=True)
    assert pinned == set()


# ── Flag OFF -> byte-identical legacy behavior ──────────────────────────────────

def test_flag_off_pins_purely_on_confidence():
    # Legacy: a name-only candidate at conf 0.95 IS pinned when the guard is off.
    resolution = {
        "vendor": [
            _cand("OTA_VENDOR_SUPPLIES.parquet", 0.95, corroborated=False),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=False)
    assert pinned == {"OTA_VENDOR_SUPPLIES.parquet"}


def test_flag_off_keeps_all_at_threshold():
    # Legacy union semantics: every candidate >= threshold is pinned.
    resolution = {
        "vendor": [
            _cand("OE_SO_HOLDS_ALL.parquet", 0.95, corroborated=False),
            _cand("OE_ORDER_HEADERS_ALL.parquet", 0.90, corroborated=True),
            _cand("noise.parquet", 0.40, corroborated=False),
        ]
    }
    pinned = select_pinnable_blobs(resolution, PIN_THRESHOLD, guard_enabled=False)
    assert pinned == {"OE_SO_HOLDS_ALL.parquet", "OE_ORDER_HEADERS_ALL.parquet"}
