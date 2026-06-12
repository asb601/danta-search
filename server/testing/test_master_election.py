"""Pure unit tests for canonical-master election (no DB, no LLM).

Group shapes mirror the live demo container 9a759559 (OEBS dump): the 'invoice'
label group has ~35 AP_* members with IDENTICAL structural stats (rows=500,
unique_rate=1.0) — only good_for text + name bareness discriminate. Election
must be abstain-biased: decisive winner or NOTHING.
"""
from __future__ import annotations

from app.services.semantic_master_election import (
    MemberEvidence,
    elect_master,
)
from app.services.semantic_policy import get_semantic_policy


def member(
    file_id: str,
    file_name: str,
    good_for: list[str],
    description: str = "",
    polarity: str | None = "vendor",
    fingerprint: str | None = None,
    row_count: int = 500,
    key_cardinality: int = 500,
) -> MemberEvidence:
    return MemberEvidence(
        file_id=file_id,
        file_name=file_name,
        entity_label="invoice",
        good_for=tuple(good_for),
        description=description,
        row_count=row_count,
        polarity=polarity,
        schema_fingerprint=fingerprint,
        key_cardinality=key_cardinality,
    )


INVOICE_GF = [
    "What is the total outstanding amount for invoices by vendor, using INVOICE_AMOUNT?",
    "How many invoices are approved versus pending, using APPROVAL_STATUS?",
    "What is the payment status of invoices due in a date range?",
]
LINES_GF = [
    "What is the line-level breakdown of invoice amounts by line type?",
    "Which invoice lines carry the highest tax amounts?",
]
TERMS_GF = [
    "What payment terms apply to vendor agreements?",
    "Which term names have the longest due-date offsets?",
]
BANK_GF = [
    "Which bank accounts are configured for payments?",
    "What are the account number formats by branch?",
]


def test_elects_the_bare_goodfor_centric_header():
    """AP_INVOICES_ALL: only bare name + fully invoice-centric good_for → elected."""
    group = [
        member("f-hdr", "AP_INVOICES_ALL", INVOICE_GF,
               description="The authoritative ledger of accounts payable invoices.",
               fingerprint="fp-twin"),
        member("f-lin", "AP_INVOICE_LINES_ALL", LINES_GF, fingerprint="fp-twin"),
        member("f-trm", "AP_TERMS_LINES", TERMS_GF),
        member("f-bnk", "AP_BANK_ACCOUNTS_ALL", BANK_GF),
    ]
    d = elect_master("invoice", group, get_semantic_policy())
    assert d.elected_file_id == "f-hdr"
    assert d.reason == "decisive"


def test_two_bare_candidates_abstain():
    """Two members both reduce to the bare concept → ambiguous → NOTHING.

    Affixes for this 3-member group (ceil(0.6*3)=2 sharers): {ap, all} — so
    AP_INVOICES_ALL and AP_INVOICES both reduce to {invoice} → 2 bare.
    """
    group = [
        member("f-a", "AP_INVOICES_ALL", INVOICE_GF),
        member("f-b", "AP_INVOICES", INVOICE_GF),
        member("f-c", "AP_INVOICE_LINES_ALL", LINES_GF),
    ]
    d = elect_master("invoice", group, get_semantic_policy())
    assert d.elected_file_id is None
    assert "bare" in d.reason


def test_no_goodfor_eligibility_abstains():
    """No member's good_for is ABOUT the concept → NOTHING (never guess by name)."""
    group = [
        member("f-a", "AP_INVOICES_ALL", BANK_GF, description=""),
        member("f-b", "AP_INVOICE_LINES_ALL", TERMS_GF),
    ]
    d = elect_master("invoice", group, get_semantic_policy())
    assert d.elected_file_id is None
    assert "eligible" in d.reason


def test_opposite_polarity_rival_abstains():
    """A reliable opposite-polarity rival inside master_twin_margin of the
    winner forces abstention even when the winner clears master_margin.

    Score math (weights 0.65/0.25/0.10, all key shares 1.0):
      winner  AP_INVOICES_ALL          cov 4/5=0.8, bare  → 0.65*0.8+0.25+0.10 = 0.87
      rival   AP_INVOICES_HISTORY_ALL  cov 1.0, not bare  → 0.65+0.10        = 0.75
      gap 0.12: >= master_margin (0.10) but < master_twin_margin (0.15) → abstain.
    """
    winner_gf = INVOICE_GF + [
        "What is the average invoice amount for a specific invoice type?",
        BANK_GF[0],  # one off-topic entry → coverage 4/5
    ]
    group = [
        member("f-ap", "AP_INVOICES_ALL", winner_gf, polarity="vendor"),
        member("f-ar", "AP_INVOICES_HISTORY_ALL", INVOICE_GF, polarity="customer"),
        member("f-lin", "AP_INVOICE_LINES_ALL", LINES_GF, polarity="vendor"),
    ]
    d = elect_master("invoice", group, get_semantic_policy())
    assert d.elected_file_id is None
    assert "polarity" in d.reason


def test_singleton_group_elects_when_about_its_concept():
    g = [member("f-1", "AP_INVOICES_ALL", INVOICE_GF)]
    d = elect_master("invoice", g, get_semantic_policy())
    assert d.elected_file_id == "f-1"


def test_singleton_group_abstains_when_goodfor_is_off_topic():
    g = [member("f-1", "AP_INVOICES_ALL", BANK_GF)]
    d = elect_master("invoice", g, get_semantic_policy())
    assert d.elected_file_id is None


def test_scores_are_reported_for_every_member():
    group = [
        member("f-hdr", "AP_INVOICES_ALL", INVOICE_GF),
        member("f-lin", "AP_INVOICE_LINES_ALL", LINES_GF),
    ]
    d = elect_master("invoice", group, get_semantic_policy())
    assert len(d.scores) == 2
    for s in d.scores:
        assert {"file_id", "file_name", "score", "coverage", "bare", "eligible"} <= set(s)
