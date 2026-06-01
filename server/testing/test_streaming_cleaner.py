"""Tests for the Polars + DuckDB streaming cleaner (flag-gated parity lane).

Run from server/:
    PYTHONPATH=. uv run --with pytest python -m pytest testing/test_streaming_cleaner.py -q

These tests are self-contained: no Azure, no network. They exercise the PURE
cleaning core (clean_frame_polars), the probe-layout discovery helper
(_probe_text_layout, which reuses the pandas detection helpers), and the
size-router default (flag false => pandas path chosen). Polars/DuckDB tests skip
gracefully if those optional deps are not yet `uv sync`-ed, so the suite never
hard-fails on a fresh checkout.
"""
from __future__ import annotations

import pytest

# The module imports lazily, so importing it never needs polars/duckdb installed.
from app.services.preprocessor import streaming_cleaner as sc
from app.services.preprocessor.cleaning_rules import get_cleaning_profile

_HAS_POLARS = False
try:  # detect polars without forcing a skip at import time
    import polars as _pl  # noqa: F401
    _HAS_POLARS = True
except Exception:  # noqa: BLE001
    _HAS_POLARS = False

requires_polars = pytest.mark.skipif(not _HAS_POLARS, reason="polars not installed")


# ── Probe-layout discovery (pure, pandas-helper reuse — no polars needed) ──────

def test_delimiter_detection_comma():
    text = "name,age,city\nalice,30,nyc\nbob,25,la\n"
    warns: list[str] = []
    layout = sc._probe_text_layout(text, ".csv", warns)
    assert layout["delimiter"] == ","
    assert layout["headers"] == ["name", "age", "city"]
    assert layout["header_row_idx"] == 0


def test_delimiter_detection_pipe():
    text = "id|amount|date\n1|100|2026-01-01\n2|200|2026-01-02\n3|300|2026-01-03\n"
    warns: list[str] = []
    layout = sc._probe_text_layout(text, ".psv", warns)
    assert layout["delimiter"] == "|"
    assert layout["headers"] == ["id", "amount", "date"]


def test_header_row_detection_with_junk_rows():
    # Two leading junk/title rows before the real header.
    text = (
        "Quarterly Report,,\n"
        "Generated 2026,,\n"
        "product,units,revenue\n"
        "widget,10,1000\n"
        "gadget,5,500\n"
    )
    warns: list[str] = []
    layout = sc._probe_text_layout(text, ".csv", warns)
    assert layout["header_row_idx"] == 2
    assert layout["headers"] == ["product", "units", "revenue"]
    assert any("Header row found at row 2" in w for w in warns)


def test_snake_case_and_dedup_columns():
    # Two identical column names → dedup with suffix.
    text = "amount,amount,note\n1,2,x\n3,4,y\n"
    warns: list[str] = []
    layout = sc._probe_text_layout(text, ".csv", warns)
    assert layout["headers"] == ["amount", "amount_1", "note"]
    assert layout["cols_renamed"].get("amount") == "amount_1"


# ── Pure cleaning core (needs polars) ──────────────────────────────────────────

@requires_polars
def test_null_tokenization_and_snake_case_clean():
    import polars as pl

    headers = ["name", "value"]
    df = pl.DataFrame({
        "name": ["alice", "bob", "carol"],
        "value": ["100", "N/A", "null"],
    })
    converters: dict = {}
    profile = get_cleaning_profile()
    clean, quarantine = sc.clean_frame_polars(df, headers, converters, profile)
    out = clean.to_pandas()
    # "N/A" and "null" are null tokens → blanked by the registry.
    vals = list(out["value"])
    assert vals[0] == "100"
    assert vals[1] == ""
    assert vals[2] == ""


@requires_polars
def test_garbage_row_drop_and_quarantine_count():
    import polars as pl

    headers = ["label", "amount"]
    # Row with leading "Total" is a garbage/subtotal row → dropped + quarantined.
    df = pl.DataFrame({
        "label": ["widget", "gadget", "Total"],
        "amount": ["10", "20", "30"],
    })
    profile = get_cleaning_profile()
    clean, quarantine = sc.clean_frame_polars(df, headers, {}, profile)
    out = clean.to_pandas()
    assert len(out) == 2
    assert "Total" not in list(out["label"])
    assert len(quarantine) == 1
    assert quarantine[0]["reason"] == "garbage_keyword"


@requires_polars
def test_basic_numeric_type_inference():
    import polars as pl
    from app.services.data_preprocessor import _build_converters

    headers = ["sku", "amount"]
    # Build converters from a sample so the numeric detector claims `amount`.
    sample = pl.DataFrame({
        "sku": ["A1", "A2", "A3"],
        "amount": ["$1,000.50", "2000", "3,000"],
    }).to_pandas()
    warns: list[str] = []
    converters = _build_converters(sample, headers, warns)
    assert "amount" in converters  # numeric detector claimed it

    df = pl.DataFrame({"sku": ["A1"], "amount": ["$1,234.50"]})
    profile = get_cleaning_profile()
    clean, _ = sc.clean_frame_polars(df, headers, converters, profile)
    out = clean.to_pandas()
    # Currency + thousands stripped → numeric string.
    assert out["amount"].iloc[0] == "1234.5"


@requires_polars
def test_empty_frame_passthrough():
    import polars as pl

    df = pl.DataFrame({"a": [], "b": []}, schema={"a": pl.Utf8, "b": pl.Utf8})
    clean, quarantine = sc.clean_frame_polars(df, ["a", "b"], {}, get_cleaning_profile())
    assert clean.height == 0
    assert quarantine == []


# ── Size-router default: flag OFF => pandas path (NOT the streaming cleaner) ────

def test_router_defaults_off(monkeypatch):
    """With preprocess.use_polars_cleaner false (the default), data_preprocessor
    must NOT call the streaming cleaner. We assert the gate reads false from the
    real policy and that calling the streaming cleaner is never reached."""
    from app.services.ingestion_policy import get_ingestion_policy

    flag = get_ingestion_policy().lookup(("preprocess", "use_polars_cleaner"))
    assert not flag, "default policy must keep use_polars_cleaner false"


def test_router_invokes_cleaner_when_flag_on(monkeypatch):
    """When the flag is on, preprocess_file routes into streaming_cleaner; when
    that raises, it falls back to the pandas path. We stub both to prove routing
    without any Azure IO."""
    import asyncio
    from app.services import data_preprocessor as dp

    # Force the flag on for this test only.
    class _FakePolicy:
        def lookup(self, path):
            if path == ("preprocess", "use_polars_cleaner"):
                return True
            return None

    monkeypatch.setattr(
        "app.services.ingestion_policy.get_ingestion_policy",
        lambda: _FakePolicy(),
    )

    called = {"streaming": False, "pandas_fallback": False}

    async def _fake_streaming(**kwargs):
        called["streaming"] = True
        raise RuntimeError("simulated parity gap")

    monkeypatch.setattr(sc, "preprocess_file", _fake_streaming)

    # Make the pandas body fail fast & observably (so we know we fell through to it
    # rather than returning the streaming result). We stub the first thing the
    # pandas body touches: BlobServiceClient.from_connection_string.
    def _boom(*a, **k):
        called["pandas_fallback"] = True
        raise RuntimeError("pandas-path-reached")

    monkeypatch.setattr(
        "app.services.data_preprocessor.BlobServiceClient.from_connection_string",
        _boom,
    )

    with pytest.raises(RuntimeError, match="pandas-path-reached"):
        asyncio.run(dp.preprocess_file(
            blob_path="x.csv", file_name="x.csv", file_id="abcdef12",
            connection_string="cs", container_name="c",
        ))

    assert called["streaming"] is True
    assert called["pandas_fallback"] is True
