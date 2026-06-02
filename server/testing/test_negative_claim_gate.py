"""Pure tests for the enforced negative-claim gate (Fix 4, no DB)."""
from __future__ import annotations

from app.services.erp.negative_claim_gate import evaluate_negative_claim


class _StubIdent:
    def __init__(self, members):
        self.member_file_ids = tuple(members)
        self.canonical_id = members[0]


class _StubMap:
    def __init__(self, table_members):
        self._t = table_members

    def resolve_table(self, name):
        return _StubIdent(self._t[name])


def test_non_negative_answer_is_noop():
    v = evaluate_negative_claim(answer="Total spend was 4.2B.", store={}, file_identities=None)
    assert not v.is_negative_claim and v.proven


def test_negative_unproven_when_partial_coverage():
    # claim says missing, but only 1 of 3 partitions scanned
    store = {"sql_attempts": [{"status": "executed", "referenced_tables": ["PROC"],
                               "referenced_file_ids": ["f1"], "logical_sql": "SELECT * FROM PROC"}]}
    fm = _StubMap({"PROC": ["f1", "f2", "f3"]})
    v = evaluate_negative_claim(answer="2023 is missing entirely.", store=store, file_identities=fm)
    assert v.is_negative_claim and not v.proven and not v.coverage_complete


def test_negative_unproven_when_date_filter_not_diagnosed():
    # full coverage, but a date-filtered empty result with no MIN/MAX probe
    store = {"sql_attempts": [{"status": "executed", "referenced_tables": ["PROC"],
                               "referenced_file_ids": ["f1", "f2"],
                               "logical_sql": "SELECT * FROM PROC WHERE year = 2025"}]}
    fm = _StubMap({"PROC": ["f1", "f2"]})
    v = evaluate_negative_claim(answer="No records found.", store=store, file_identities=fm)
    assert v.is_negative_claim and not v.proven
    assert "date_window_overlap" in v.missing_diagnostics


def test_negative_proven_with_coverage_and_distinct_probe():
    store = {"sql_attempts": [
        {"status": "executed", "referenced_tables": ["PROC"], "referenced_file_ids": ["f1", "f2"],
         "logical_sql": "SELECT DISTINCT status FROM PROC"},
        {"status": "executed", "referenced_tables": ["PROC"], "referenced_file_ids": ["f1", "f2"],
         "logical_sql": "SELECT * FROM PROC WHERE status = 'Shipped'"},
    ]}
    fm = _StubMap({"PROC": ["f1", "f2"]})
    v = evaluate_negative_claim(answer="There are no shipped records.", store=store, file_identities=fm)
    assert v.is_negative_claim and v.coverage_complete and v.diagnosed and v.proven


def test_never_raises_on_malformed_store():
    v = evaluate_negative_claim(answer="no data", store={"sql_attempts": None}, file_identities=None)
    assert v.is_negative_claim and not v.proven
