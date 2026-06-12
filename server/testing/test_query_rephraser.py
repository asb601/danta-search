"""Tests for the query rephrasing layer (services/query_rephraser.py).

Verifies the safety contract: clarification replaces the query only when the
rephrase is sane; every failure mode (disabled flag, LLM error, empty/over-long
output) falls back to the ORIGINAL query so the ask can never be corrupted.
"""
import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from app.services import query_rephraser
from app.services.query_rephraser import rephrase_query, _is_runaway


def _run(coro):
    return asyncio.run(coro)


def _fake_client(returns: str):
    """A stand-in Azure client whose chat.completions.create returns `returns`."""
    msg = SimpleNamespace(content=returns)
    choice = SimpleNamespace(message=msg)
    resp = SimpleNamespace(choices=[choice])

    class _Completions:
        def create(self, **kwargs):
            return resp

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))
    return client, "gpt-4o-mini-deploy"


def _settings(enabled: bool = True):
    return SimpleNamespace(
        QUERY_REPHRASE_ENABLED=enabled,
        AZURE_OPENAI_DEPLOYMENT_MINI="gpt-4o-mini-deploy",
    )


def test_disabled_flag_returns_original():
    with patch.object(query_rephraser, "get_settings", lambda: _settings(enabled=False)):
        r = _run(rephrase_query("show me reveune by vndr"))
    assert r.text == "show me reveune by vndr"
    assert r.changed is False
    assert r.reason == "disabled"


def test_empty_input_returns_original():
    with patch.object(query_rephraser, "get_settings", lambda: _settings()):
        r = _run(rephrase_query("   "))
    assert r.changed is False
    assert r.reason == "empty_input"


def test_messy_prompt_is_rephrased():
    # ERP-enriched rewrite naming tables/joins must pass through (no relative cap).
    erp = (
        "Show total open AP invoice amount by supplier: join AP_INVOICES_ALL to "
        "AP_SUPPLIERS on VENDOR_ID, filter PAYMENT_STATUS_FLAG = 'N', sum AMOUNT."
    )
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client(erp)):
        r = _run(rephrase_query("show me reveune by vndr"))
    assert r.text == erp
    assert r.changed is True
    assert r.reason == "rephrased"


def test_clean_prompt_returns_unchanged():
    q = "What is total revenue by vendor?"
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client(q)):
        r = _run(rephrase_query(q))
    assert r.text == q
    assert r.changed is False
    assert r.reason == "unchanged"


def test_llm_error_falls_back_to_original():
    def _boom():
        raise RuntimeError("azure down")

    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", _boom):
        r = _run(rephrase_query("show me reveune by vndr"))
    assert r.text == "show me reveune by vndr"
    assert r.changed is False
    assert r.reason == "error"


def test_empty_llm_output_falls_back_to_original():
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client("   ")):
        r = _run(rephrase_query("show me reveune by vndr"))
    assert r.text == "show me reveune by vndr"
    assert r.changed is False
    assert r.reason == "empty"


def test_runaway_rephrase_falls_back_to_original():
    original = "top vendors"
    runaway = "x" * 6500  # past the 6000-char runaway cap → model rambled/answered
    assert _is_runaway(runaway) is True
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client(runaway)):
        r = _run(rephrase_query(original))
    assert r.text == original
    assert r.changed is False
    assert r.reason == "too_long"


def _capturing_client(returns: str, sink: list):
    """Like _fake_client, but records the kwargs passed to create() into `sink`."""
    msg = SimpleNamespace(content=returns)
    resp = SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    class _Completions:
        def create(self, **kwargs):
            sink.append(kwargs)
            return resp

    return SimpleNamespace(chat=SimpleNamespace(completions=_Completions())), "dep"


def test_domain_is_injected_into_prompt():
    sink: list = []
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _capturing_client("From X return a", sink)):
        _run(rephrase_query("open receivables", domain="SAPDATA EBS"))
    prompt = sink[0]["messages"][0]["content"]
    assert "SAPDATA EBS" in prompt  # the model is told which system to target


def test_domain_absent_is_not_in_prompt():
    sink: list = []
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _capturing_client("From X return a", sink)):
        _run(rephrase_query("open receivables", domain=None))
    prompt = sink[0]["messages"][0]["content"]
    assert "target source system" not in prompt.lower()


def test_domain_is_appended_in_code():
    # The domain comes from the caller (selected domain filter), not the LLM.
    rewrite = "From AP_HOLDS_ALL ..., return invoice_num, filtering release_date IS NULL"
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client(rewrite)):
        r = _run(rephrase_query("invoices on hold", domain="OEBS"))
    assert r.text.endswith("-- OEBS")
    assert r.changed is True


def test_domain_append_is_idempotent():
    # If the model already emitted the marker, we must not duplicate it.
    rewrite = "From AP_HOLDS_ALL ..., filtering release_date IS NULL -- OEBS"
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client(rewrite)):
        r = _run(rephrase_query("invoices on hold", domain="OEBS"))
    assert r.text.count("-- OEBS") == 1


def test_no_domain_means_no_marker():
    rewrite = "From AP_HOLDS_ALL ..., filtering release_date IS NULL"
    with patch.object(query_rephraser, "get_settings", lambda: _settings()), \
         patch.object(query_rephraser, "get_client", lambda: _fake_client(rewrite)):
        r = _run(rephrase_query("invoices on hold", domain=None))
    assert "--" not in r.text


def test_runaway_guard_allows_erp_enriched_rewrite():
    # A long, table/join-naming ERP rewrite is expected and must NOT be rejected.
    enriched = (
        "Return total open Accounts Payable invoice amount per supplier by joining "
        "AP_INVOICES_ALL.VENDOR_ID to AP_SUPPLIERS.VENDOR_ID, filtering "
        "PAYMENT_STATUS_FLAG = 'N', grouping by AP_SUPPLIERS.VENDOR_NAME."
    )
    assert _is_runaway(enriched) is False
