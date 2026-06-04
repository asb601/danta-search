"""Pure tests for the single tunables/score-logging source (Spec §3 invariant 4)."""
from __future__ import annotations

from pdf_chat.tunables import get_tunable, log_gate_decision, TUNABLE_DEFAULTS


def test_get_tunable_returns_explicit_default_when_unset():
    # No env, no DB override → the caller-supplied default wins.
    assert get_tunable("container-1", "made_up_key", 0.42) == 0.42


def test_get_tunable_env_override_beats_default(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_CONTEXT_TOKEN_BUDGET", "1234")
    assert get_tunable("container-1", "context_token_budget", 8000) == 1234


def test_get_tunable_env_float_coerced(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_DIGITAL_TEXT_COVERAGE", "0.55")
    assert get_tunable("c", "digital_text_coverage", 0.7) == 0.55


def test_get_tunable_env_bad_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("PDF_TUNABLE_RERANK_TOP_N", "not-an-int")
    assert get_tunable("c", "rerank_top_n", 12) == 12


def test_tunable_defaults_registry_is_a_dict():
    assert isinstance(TUNABLE_DEFAULTS, dict)
    assert "context_token_budget" in TUNABLE_DEFAULTS


def test_infra_marker_is_registered(pytestconfig):
    markers = pytestconfig.getini("markers")
    assert any(m.startswith("infra") for m in markers)


def test_log_gate_decision_returns_structured_record():
    rec = log_gate_decision(
        "digital_vs_scanned",
        score=0.83,
        threshold=0.70,
        outcome="digital",
        container_id="c-1",
        page_num=3,
    )
    assert rec["gate"] == "digital_vs_scanned"
    assert rec["score"] == 0.83
    assert rec["threshold"] == 0.70
    assert rec["outcome"] == "digital"
    assert rec["page_num"] == 3
    assert rec["passed"] is True  # score >= threshold


def test_log_gate_decision_passed_false_when_below_threshold():
    rec = log_gate_decision("rerank_skip", score=2, threshold=4, outcome="skip")
    assert rec["passed"] is False


def test_db_lookup_override_beats_env(monkeypatch):
    from pdf_chat import tunables

    monkeypatch.setenv("PDF_TUNABLE_RERANK_TOP_N", "8")
    tunables.set_db_lookup(lambda cid, key: "5" if key == "rerank_top_n" else None)
    try:
        assert tunables.get_tunable("c-9", "rerank_top_n", 12) == 5
    finally:
        tunables.set_db_lookup(None)


def test_tunable_model_table_name():
    from pdf_chat.models.tunable import PdfGraphRagTunable

    assert PdfGraphRagTunable.__tablename__ == "pdf_graphrag_tunables"
