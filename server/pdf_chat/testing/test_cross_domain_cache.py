"""Phase 4 Task 6 — cross-domain cache invalidation.

A cross-domain answer (PDF entity → value-evidenced bridge → ``structured_query``
against the CSV runtime) depends on BOTH knowledge bases. Its cache key must
therefore fold in a version stamp for the CSV semantic layer AND the PDF graph
extraction, so that bumping either side transparently evicts every stale
cross-domain answer. A pure-PDF answer (``structured_query`` not used) keeps the
plain base key untouched.
"""
from __future__ import annotations

from pdf_chat.agent.cross_domain_cache import (
    build_cross_domain_cache_key,
    version_stamps,
)


def test_not_cross_domain_equals_base_key():
    base = "abc123"
    out = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key=base,
        structured_query_used=False,
        version_stamps={"csv_semantic_layer": "7", "graph_extraction": "3"},
    )
    assert out == base


def test_cross_domain_differs_from_base_key():
    base = "abc123"
    out = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key=base,
        structured_query_used=True,
        version_stamps={"csv_semantic_layer": "7", "graph_extraction": "3"},
    )
    assert out != base


def test_different_csv_semantic_version_changes_key():
    base = "abc123"
    k1 = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key=base,
        structured_query_used=True,
        version_stamps={"csv_semantic_layer": "7", "graph_extraction": "3"},
    )
    k2 = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key=base,
        structured_query_used=True,
        version_stamps={"csv_semantic_layer": "8", "graph_extraction": "3"},
    )
    assert k1 != k2


def test_different_graph_extraction_version_changes_key():
    base = "abc123"
    k1 = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key=base,
        structured_query_used=True,
        version_stamps={"csv_semantic_layer": "7", "graph_extraction": "3"},
    )
    k2 = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key=base,
        structured_query_used=True,
        version_stamps={"csv_semantic_layer": "7", "graph_extraction": "4"},
    )
    assert k1 != k2


def test_same_inputs_are_deterministic():
    stamps = {"csv_semantic_layer": "7", "graph_extraction": "3"}
    k1 = build_cross_domain_cache_key(
        tenant_id="t1", base_key="abc", structured_query_used=True, version_stamps=stamps
    )
    k2 = build_cross_domain_cache_key(
        tenant_id="t1", base_key="abc", structured_query_used=True, version_stamps=stamps
    )
    assert k1 == k2


def test_missing_stamps_default_to_zero_not_crash():
    # Both stamps absent: must not KeyError, both default to "0".
    out = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key="abc",
        structured_query_used=True,
        version_stamps={},
    )
    assert out != "abc"
    # Explicit "0" stamps must produce the SAME key as missing stamps.
    explicit_zero = build_cross_domain_cache_key(
        tenant_id="t1",
        base_key="abc",
        structured_query_used=True,
        version_stamps={"csv_semantic_layer": "0", "graph_extraction": "0"},
    )
    assert out == explicit_zero


def test_version_stamps_returns_dict_with_required_keys_no_infra():
    # Best-effort read with no DB/infra: must return a dict carrying both keys,
    # each defaulting to "0" when the stamp source is unavailable.
    stamps = version_stamps("tenant-without-infra")
    assert isinstance(stamps, dict)
    assert stamps["csv_semantic_layer"] == "0"
    assert stamps["graph_extraction"] == "0"


def test_version_stamps_does_not_import_app():
    # The reader must not pull anything under server/app/ into sys.modules.
    import sys

    before = {m for m in sys.modules if m == "app" or m.startswith("app.")}
    version_stamps("t1")
    after = {m for m in sys.modules if m == "app" or m.startswith("app.")}
    assert after == before
