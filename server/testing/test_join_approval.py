"""Pure tests for join-approval classification (relational fix, no DB).

The promotion gate is VALUE + IDENTITY driven and ROLE-INDEPENDENT. The decisive
new separator is the SAME-KEY IDENTITY GUARD: an edge is only promotable when its
two join columns are the SAME business key (normalize(left) == normalize(right),
e.g. CUSTOMER_ID == CUSTOMER_ID). This kills the dominant real-DB failure mode —
two independent small-integer id sequences that coincidentally value-overlap
(BANK_ACCOUNT_ID == VENDOR_ID, PLAN_ID == COST_TYPE_ID) — which the previous
single-side role + confidence-floor rule wrongly approved.

Promote an edge to `approved` IFF ALL hold (role-independent, no confidence floor):
  1. normalize(left_column) == normalize(right_column)  (same business key)
  2. value_overlap >= policy.min_join_overlap            (0.50)
  3. min_cardinality >= policy.min_join_cardinality      (8)
  4. ubiquity <= policy.ubiquity_ceiling                 (0.60)

Numbers below are the SHAPES of the REAL edges replayed from 197,004 production
edges — used as fixtures only. No column-name lists in the code under test; the
guard is a pure string-equality check on the two threaded column NAMES.

Run: cd server && uv run python -m pytest testing/test_join_approval.py -q
"""
from __future__ import annotations

import pytest

from app.services.semantic_layer_builder import classify_join_approval
from app.services.semantic_policy import get_semantic_policy

P = get_semantic_policy()

# A value-validated business entity key (CUSTOMER_ID / VENDOR_ID shape).
ENTITY_KEY = "custom:entity_key:customer"
# A reference key that is risky single-column by construction.
REFERENCE_KEY = "custom:reference_key:customer"
# A reference key for the doc/document shape.
REFERENCE_KEY_DOC = "custom:reference_key:doc"


# ───────────────────────── MUST APPROVE (real masters) ─────────────────────────

def test_customer_id_same_name_reference_role_low_confidence_is_approved():
    # CUSTOMER_ID == CUSTOMER_ID, REAL numbers: overlap 0.64, card 312,
    # ubiquity 0.27, confidence 0.69 — and a reference_key:customer role (which
    # the risky-single-column gate previously killed). The same-key guard +
    # value evidence must APPROVE it DESPITE low confidence AND the reference role.
    status, reason = classify_join_approval(
        role=REFERENCE_KEY,
        relationship_type="many_to_many",
        value_overlap=0.64,
        confidence=0.69,
        cardinality_left=312,
        cardinality_right=312,
        ubiquity=0.27,
        has_companion=False,
        policy=P,
        left_column="CUSTOMER_ID",
        right_column="CUSTOMER_ID",
    )
    assert status == "approved"
    assert reason is None


def test_vendor_id_same_name_is_approved():
    # VENDOR_ID == VENDOR_ID, REAL numbers: overlap 0.94, card 190, ubiquity 0.13.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.94,
        confidence=0.80,
        cardinality_left=190,
        cardinality_right=190,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="VENDOR_ID",
        right_column="VENDOR_ID",
    )
    assert status == "approved"
    assert reason is None


def test_same_name_guard_normalizes_case_and_whitespace():
    # normalize = upper/strip: " customer_id " == "CUSTOMER_ID".
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.94,
        confidence=0.80,
        cardinality_left=190,
        cardinality_right=190,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column=" customer_id ",
        right_column="CUSTOMER_ID",
    )
    assert status == "approved"
    assert reason is None


# ─────────────────────── MUST REJECT (cross-name false joins) ───────────────────

def test_bank_account_id_vs_vendor_id_cross_name_is_candidate():
    # THE DOMINANT FALSE JOIN: two independent small-int id sequences that
    # coincidentally value-overlap. REAL numbers: overlap 0.81, card 181,
    # ubiquity 0.13, confidence 0.80 — passes EVERY value gate, but the names
    # differ, so the same-key identity guard must hold it as a candidate.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.81,
        confidence=0.80,
        cardinality_left=181,
        cardinality_right=181,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="BANK_ACCOUNT_ID",
        right_column="VENDOR_ID",
    )
    assert status == "candidate"
    assert reason is not None


def test_plan_id_vs_cost_type_id_cross_name_is_candidate():
    # Another coincidental cross-name overlap (PLAN_ID == COST_TYPE_ID).
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.75,
        confidence=0.78,
        cardinality_left=120,
        cardinality_right=120,
        ubiquity=0.10,
        has_companion=False,
        policy=P,
        left_column="PLAN_ID",
        right_column="COST_TYPE_ID",
    )
    assert status == "candidate"
    assert reason is not None


def test_created_by_same_name_but_ubiquitous_is_candidate():
    # CREATED_BY == CREATED_BY: SAME name, but ubiquity 1.0 (present in ~all
    # files). The ubiquity ceiling must override the same-name match.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=1.0,
        confidence=0.85,
        cardinality_left=200,
        cardinality_right=200,
        ubiquity=1.0,
        has_companion=False,
        policy=P,
        left_column="CREATED_BY",
        right_column="CREATED_BY",
    )
    assert status == "candidate"
    assert reason is not None
    assert "audit" in reason or "ubiquit" in reason


def test_org_id_same_name_but_low_cardinality_is_candidate():
    # ORG_ID == ORG_ID: SAME name, but cardinality 1 — degenerate key.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=1.0,
        confidence=0.85,
        cardinality_left=1,
        cardinality_right=1,
        ubiquity=0.10,
        has_companion=False,
        policy=P,
        left_column="ORG_ID",
        right_column="ORG_ID",
    )
    assert status == "candidate"
    assert reason is not None
    assert "cardinality" in reason


def test_same_name_low_overlap_is_candidate():
    # SAME name, real cardinality, low ubiquity, but no value reconciliation.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.01,
        confidence=0.85,
        cardinality_left=500,
        cardinality_right=500,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="CUSTOMER_ID",
        right_column="CUSTOMER_ID",
    )
    assert status == "candidate"
    assert reason is not None
    assert "overlap" in reason


def test_cross_name_falls_through_to_risky_single_column_gate():
    # A risky single-column reference role on a CROSS-name edge is NOT promoted
    # (the guard requires same names) and is held by the risky-single-column gate.
    status, reason = classify_join_approval(
        role=REFERENCE_KEY_DOC,
        relationship_type="many_to_many",
        value_overlap=1.0,
        confidence=0.90,
        cardinality_left=500,
        cardinality_right=500,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="INVOICE_DOC_ID",
        right_column="RECEIPT_DOC_ID",
    )
    assert status == "candidate"
    assert reason is not None


def test_missing_column_names_cannot_promote():
    # Without column NAMES the identity guard cannot fire, so a strong edge is
    # held as a candidate (fail-safe: never promote on absent identity evidence).
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=1.0,
        confidence=0.85,
        cardinality_left=500,
        cardinality_right=500,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column=None,
        right_column=None,
    )
    assert status == "candidate"
    assert reason is not None


def test_ubiquity_ceiling_exists_on_policy():
    # The threshold must be a real, env-overridable policy field.
    assert hasattr(P, "ubiquity_ceiling")
    assert 0.0 < P.ubiquity_ceiling < 1.0


# ─────────────── CLONE GUARD (templated/copied column, not a real FK) ───────────
#
# A copied/templated column (not a referential FK) has the verified signature:
#   value_overlap >= ~0.999  AND  cardinality_left == cardinality_right
# (the two columns hold the IDENTICAL generated value set, so both cardinalities
# equal min_cardinality). Real master keys never have this — their two sides
# differ in cardinality. The guard runs INSIDE the same-name promotion branch:
# a clone same-name edge becomes a candidate; a real master same-name edge still
# approves. Numbers below are the SHAPES of REAL edges replayed from production.


def test_po_header_id_clone_same_name_is_candidate():
    # THE 32.5%-of-approvals FALSE JOIN: AP_BATCHES_ALL ⋈ AP_BANK_ACCOUNTS_ALL on
    # PO_HEADER_ID == PO_HEADER_ID at overlap 1.000 with card_left==card_right==50
    # — an identical COPIED value set between unrelated tables, not a real FK.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=1.000,
        confidence=0.85,
        cardinality_left=50,
        cardinality_right=50,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="PO_HEADER_ID",
        right_column="PO_HEADER_ID",
    )
    assert status == "candidate"
    assert reason is not None
    assert "templated" in reason or "copied" in reason


def test_customer_id_differing_cardinality_is_not_clone_and_approves():
    # CUSTOMER_ID == CUSTOMER_ID, REAL numbers: overlap 0.64, cards 312/300
    # (differ → NOT clone). The real master must still APPROVE.
    status, reason = classify_join_approval(
        role=REFERENCE_KEY,
        relationship_type="many_to_many",
        value_overlap=0.64,
        confidence=0.69,
        cardinality_left=312,
        cardinality_right=300,
        ubiquity=0.27,
        has_companion=False,
        policy=P,
        left_column="CUSTOMER_ID",
        right_column="CUSTOMER_ID",
    )
    assert status == "approved"
    assert reason is None


def test_vendor_id_target_edge_high_overlap_differing_cardinality_approves():
    # THE REAL TARGET PATH: VENDOR_ID AP_INVOICE_LINES_ALL ⋈ PO_DISTRIBUTIONS_ALL,
    # REAL numbers overlap 0.95 with cards 181/190 (differ → NOT clone). This edge
    # MUST survive the clone guard — it is the path the runtime needs.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.95,
        confidence=0.80,
        cardinality_left=181,
        cardinality_right=190,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="VENDOR_ID",
        right_column="VENDOR_ID",
    )
    assert status == "approved"
    assert reason is None


def test_vendor_id_same_name_differing_cardinality_approves():
    # VENDOR_ID == VENDOR_ID, REAL numbers: overlap 0.94, cards 190/188 (differ).
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.94,
        confidence=0.80,
        cardinality_left=190,
        cardinality_right=188,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="VENDOR_ID",
        right_column="VENDOR_ID",
    )
    assert status == "approved"
    assert reason is None


def test_high_overlap_below_floor_but_equal_cardinality_is_not_clone():
    # EDGE: overlap 0.9995 (>= floor 0.999) but cards DIFFER → NOT clone → approve.
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.9995,
        confidence=0.80,
        cardinality_left=190,
        cardinality_right=188,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="VENDOR_ID",
        right_column="VENDOR_ID",
    )
    assert status == "approved"
    assert reason is None


def test_equal_cardinality_but_overlap_below_floor_is_not_clone():
    # EDGE: cards EQUAL (both 200) but overlap 0.80 (< floor 0.999) → NOT clone →
    # normal same-name promotion path → approve (overlap clears min_join_overlap).
    status, reason = classify_join_approval(
        role=ENTITY_KEY,
        relationship_type="many_to_many",
        value_overlap=0.80,
        confidence=0.80,
        cardinality_left=200,
        cardinality_right=200,
        ubiquity=0.13,
        has_companion=False,
        policy=P,
        left_column="VENDOR_ID",
        right_column="VENDOR_ID",
    )
    assert status == "approved"
    assert reason is None


def test_clone_overlap_floor_exists_on_policy():
    # The clone threshold must be a real, env-overridable policy field.
    assert hasattr(P, "clone_overlap_floor")
    assert 0.99 < P.clone_overlap_floor <= 1.0


