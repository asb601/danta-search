"""Pytest suite for the self-tuning ingestion config layer.

Covers:
  A. Pure formula correctness (compute_ingestion_knobs) against a fixed table.
  B. Override behaviour (ingest_rpm, reserve_cores, k_pandas).
  C. Detection robustness (cgroup v2 mocking, psutil-absent, min-of-signals).
  D. Precedence in ingestion_policy (env > non-null JSON > formula).
  E. Boot smoke (get_settings + formula-backed knobs are sane ints).

Run with:
    cd server && uv run --with pytest pytest testing/test_ingestion_resource_formula.py -v
"""

from __future__ import annotations

import importlib

import pytest

from app.services.resource_profile import (
    ResourceProfile,
    compute_ingestion_knobs,
)
import app.services.resource_profile as rp

GIB = 1024 ** 3


def _profile(cpu: int, ram_total_gib: int) -> ResourceProfile:
    """Build a profile with ram_available = round(0.85 * ram_total)."""
    ram_total = ram_total_gib * GIB
    ram_available = round(0.85 * ram_total)
    return ResourceProfile(
        cpu_count=cpu,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_free_bytes=50 * GIB,
        source="unit-test",
        is_container=False,
    )


# ---------------------------------------------------------------------------
# A. Formula correctness — the core. Pure function, no mocking.
# ---------------------------------------------------------------------------

# cpu, ram_gib, preprocess, io, celery, parquet
_FORMULA_TABLE = [
    (2, 8, 1, 16, 1, 1),
    (4, 16, 2, 32, 2, 2),
    (8, 32, 6, 64, 6, 6),
    (16, 64, 14, 64, 14, 14),
]


@pytest.mark.parametrize(
    "cpu,ram_gib,preprocess,io,celery,parquet", _FORMULA_TABLE
)
def test_formula_exact_values(cpu, ram_gib, preprocess, io, celery, parquet):
    knobs = compute_ingestion_knobs(_profile(cpu, ram_gib))
    assert knobs["preprocess_concurrency"] == preprocess
    assert knobs["io_concurrency"] == io
    assert knobs["celery_worker_concurrency"] == celery
    assert knobs["parquet_conversion_concurrency"] == parquet


@pytest.mark.parametrize("row", _FORMULA_TABLE)
def test_formula_invariants(row):
    cpu, ram_gib = row[0], row[1]
    knobs = compute_ingestion_knobs(_profile(cpu, ram_gib))
    # Every concurrency knob is a usable pool size (>= 1).
    for key in (
        "preprocess_concurrency",
        "excel_preprocess_concurrency",
        "parquet_conversion_concurrency",
        "celery_worker_concurrency",
        "io_concurrency",
        "ingestion_cores",
    ):
        assert knobs[key] >= 1, f"{key} must be >= 1, got {knobs[key]}"
    # duckdb_threads >= 1 and no ZeroDivision even when parquet pool is small.
    assert knobs["duckdb_threads"] >= 1
    # small_file_threshold strictly positive.
    assert knobs["small_file_threshold_bytes"] > 0


def test_formula_no_zero_division_tiny_box():
    # 1 cpu / 1 GiB: parquet_conversion_concurrency floors to 1, so the
    # duckdb_threads division (ingestion_cores // parquet) must not blow up.
    knobs = compute_ingestion_knobs(_profile(1, 1))
    assert knobs["parquet_conversion_concurrency"] >= 1
    assert knobs["duckdb_threads"] >= 1
    assert knobs["small_file_threshold_bytes"] > 0


# ---------------------------------------------------------------------------
# B. Overrides
# ---------------------------------------------------------------------------

def test_override_ingest_rpm_caps_io():
    p = _profile(8, 32)
    base_io = compute_ingestion_knobs(p)["io_concurrency"]
    assert base_io == 64  # uncapped formula on 8 cpu
    # rpm=120 -> floor(120/60*3.0)=6, which is < 64, so the cap bites.
    capped = compute_ingestion_knobs(p, overrides={"ingest_rpm": 120})["io_concurrency"]
    assert capped == 6


def test_override_reserve_cores_zero_raises_cores():
    """SPEC expectation: reserve_cores=0 on cpu=8 -> ingestion_cores rises to 8,
    so preprocess_concurrency increases accordingly (8 not 6)."""
    base = compute_ingestion_knobs(_profile(8, 32))
    overridden = compute_ingestion_knobs(_profile(8, 32), overrides={"reserve_cores": 0})
    assert overridden["ingestion_cores"] == 8
    assert overridden["preprocess_concurrency"] > base["preprocess_concurrency"]


def test_override_big_k_pandas_clamps_preprocess_to_one():
    # A huge per-job RAM multiplier makes the RAM ceiling allow only 1 job.
    knobs = compute_ingestion_knobs(_profile(8, 32), overrides={"k_pandas": 1000})
    assert knobs["preprocess_concurrency"] == 1


# ---------------------------------------------------------------------------
# C. Detection robustness
# ---------------------------------------------------------------------------

def test_cgroup_v2_quota_and_memory(monkeypatch):
    """cgroup v2: cpu.max '150000 100000' (=1.5 cores -> floor 1) and
    memory.max = 8 GiB. The profile must respect both and take MIN RAM."""
    rp.get_resource_profile.cache_clear()

    real_open = open
    eight_gib = str(8 * GIB)

    def fake_open(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/cpu.max":
            from io import StringIO
            return StringIO("150000 100000")
        if path == "/sys/fs/cgroup/memory.max":
            from io import StringIO
            return StringIO(eight_gib)
        if path == "/sys/fs/cgroup/memory.current":
            from io import StringIO
            return StringIO("0")
        return real_open(path, *args, **kwargs)

    # Patch the builtin used by the module's _read_int_file / cgroup readers.
    monkeypatch.setattr("builtins.open", fake_open)
    # Make psutil host RAM huge so the cgroup MIN is what actually wins.
    if rp.psutil is not None:
        class _VM:
            total = 256 * GIB
            available = 200 * GIB

        monkeypatch.setattr(rp.psutil, "virtual_memory", lambda: _VM())
        monkeypatch.setattr(rp.psutil, "cpu_count", lambda logical=True: 64)
    # Affinity (Linux) could otherwise pull cpu below the cgroup; neutralize it
    # so the cgroup rounding (1.5 -> 1) is what we observe.
    if hasattr(rp.os, "sched_getaffinity"):
        monkeypatch.setattr(rp.os, "sched_getaffinity", lambda pid: set(range(64)))
    monkeypatch.setattr(rp.os, "cpu_count", lambda: 64)

    try:
        prof = rp.get_resource_profile()
        # 1.5-core quota rounds DOWN to 1 per the documented rounding.
        assert prof.cpu_count == 1
        # RAM is the cgroup ceiling (MIN with the huge host), == 8 GiB.
        assert prof.ram_total_bytes == 8 * GIB
        assert prof.is_container is True
    finally:
        rp.get_resource_profile.cache_clear()


def test_psutil_absent_still_boots(monkeypatch):
    """psutil=None must not crash get_resource_profile (os fallback path)."""
    rp.get_resource_profile.cache_clear()
    monkeypatch.setattr(rp, "psutil", None)
    try:
        prof = rp.get_resource_profile()
        assert isinstance(prof, ResourceProfile)
        assert prof.cpu_count >= 1
        assert prof.ram_total_bytes >= 1
        assert prof.ram_available_bytes >= 0
    finally:
        rp.get_resource_profile.cache_clear()


def test_ram_total_takes_min_of_cgroup_and_host(monkeypatch):
    """Host huge, cgroup small -> ram_total must be the cgroup (MIN) value."""
    rp.get_resource_profile.cache_clear()
    real_open = open
    four_gib = str(4 * GIB)

    def fake_open(path, *args, **kwargs):
        if path == "/sys/fs/cgroup/memory.max":
            from io import StringIO
            return StringIO(four_gib)
        if path == "/sys/fs/cgroup/memory.current":
            from io import StringIO
            return StringIO("0")
        # No CPU cgroup -> let CPU detection use host signals.
        if path == "/sys/fs/cgroup/cpu.max":
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr("builtins.open", fake_open)
    if rp.psutil is not None:
        class _VM:
            total = 512 * GIB
            available = 400 * GIB

        monkeypatch.setattr(rp.psutil, "virtual_memory", lambda: _VM())

    try:
        prof = rp.get_resource_profile()
        assert prof.ram_total_bytes == 4 * GIB
    finally:
        rp.get_resource_profile.cache_clear()


# ---------------------------------------------------------------------------
# D. Precedence (integration with ingestion_policy via get_settings)
# ---------------------------------------------------------------------------

def _fresh_settings():
    """get_settings is lru_cached; clear it and the policy cache before reading."""
    from app.core.config import get_settings
    from app.services import ingestion_policy as pol

    get_settings.cache_clear()
    pol.get_ingestion_policy.cache_clear()
    return get_settings


def test_env_var_wins_over_formula(monkeypatch):
    monkeypatch.setenv("CELERY_WORKER_CONCURRENCY", "7")
    get_settings = _fresh_settings()
    try:
        assert get_settings().CELERY_WORKER_CONCURRENCY == 7
    finally:
        get_settings.cache_clear()
        from app.services import ingestion_policy as pol
        pol.get_ingestion_policy.cache_clear()


def test_formula_resolves_when_no_env_and_json_null(monkeypatch):
    monkeypatch.delenv("CELERY_WORKER_CONCURRENCY", raising=False)
    rp.get_resource_profile.cache_clear()
    get_settings = _fresh_settings()
    try:
        value = get_settings().CELERY_WORKER_CONCURRENCY
        assert isinstance(value, int)
        assert value >= 1
    finally:
        get_settings.cache_clear()
        from app.services import ingestion_policy as pol
        pol.get_ingestion_policy.cache_clear()
        rp.get_resource_profile.cache_clear()


# ---------------------------------------------------------------------------
# E. Boot smoke
# ---------------------------------------------------------------------------

def test_boot_smoke_settings_load():
    from app.core.config import get_settings

    settings = get_settings()
    assert settings is not None
    for name in (
        "CELERY_WORKER_CONCURRENCY",
        "PARQUET_CONVERSION_CONCURRENCY",
        "INGEST_PREPROCESS_CONCURRENCY",
        "REINGEST_BATCH_SIZE",
    ):
        value = getattr(settings, name)
        assert isinstance(value, int), f"{name} should be int, got {type(value)}"
        assert value >= 1, f"{name} should be >= 1, got {value}"
