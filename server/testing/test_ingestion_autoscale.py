"""Pytest suite for the ingestion autoscale decision plane.

Mirrors testing/test_ingestion_resource_formula.py in shape: the PURE decision
formula (compute_desired_capacity / pool_quota_ceiling) is exercised directly
with hand-built inputs — no Redis, no Azure, no mocking required.

Covers:
  A. Scales out under load; drain ETA honours the SLA.
  B. Clamps to max_instances.
  C. Clamps to the LLM quota ceiling (single-RPM fallback).
  D. Scale-in held until K idle ticks; scale-out ignores the idle counter.
  E. Never below min_instances; degenerate overrides never crash.
  F. pool_quota_ceiling scales linearly with #deployments and binds on the
     smaller (embedding) lane; empty pool -> None; pool clamp wins in
     compute_desired_capacity.

Run with:
    cd server && uv run pytest testing/test_ingestion_autoscale.py -q
"""

from __future__ import annotations

from collections import namedtuple

import pytest

from app.services.ingestion_autoscale import (
    ScaleDecision,
    compute_desired_capacity,
    pool_quota_ceiling,
)

# Try to use the real Deployment if model_pool exists by run time; otherwise use
# a minimal stand-in exposing exactly the attributes pool_quota_ceiling reads
# (.kind / .tpm). The ceiling math only ever touches those two fields.
try:  # pragma: no cover - depends on whether model_pool has shipped
    from app.core.model_pool import Deployment as _RealDeployment  # type: ignore

    def _dep(kind: str, tpm: int):
        return _RealDeployment(
            name=f"{kind}-{tpm}",
            kind=kind,
            endpoint="",
            deployment_id="",
            rpm=tpm,
            tpm=tpm,
            weight=1.0,
            region="",
        )
except Exception:  # noqa: BLE001
    _Deployment = namedtuple("Deployment", ["kind", "tpm"])

    def _dep(kind: str, tpm: int):
        return _Deployment(kind=kind, tpm=tpm)


# ---------------------------------------------------------------------------
# A. Scales out under load.
# ---------------------------------------------------------------------------

def test_scales_out_under_load():
    # ceil(600 * 45 / (600 * 1)) = 45 instances.
    d = compute_desired_capacity(600)
    assert isinstance(d, ScaleDecision)
    assert d.desired_instances == 45
    assert d.reason == "demand"
    assert d.quota_limited is False
    # At exactly the demand-matched capacity the backlog drains within the SLA.
    assert d.drain_eta_seconds <= 600


def test_scale_out_grows_with_backlog():
    small = compute_desired_capacity(100).desired_instances
    big = compute_desired_capacity(400).desired_instances
    assert big > small


# ---------------------------------------------------------------------------
# B. Clamp to max_instances.
# ---------------------------------------------------------------------------

def test_clamps_to_max_instances():
    d = compute_desired_capacity(100_000)
    assert d.desired_instances == 50  # default max_instances
    assert d.reason == "clamped_max"


# ---------------------------------------------------------------------------
# C. Clamp to the quota ceiling (single-RPM fallback path).
# ---------------------------------------------------------------------------

def test_clamps_to_quota_ceiling():
    # ingest_rpm=120 -> floor(120/60 * 45) = 90 in-flight files -> 90 instances.
    # max_instances raised to 500 so the quota clamp (90) is what binds.
    d = compute_desired_capacity(
        100_000, overrides={"ingest_rpm": 120, "max_instances": 500}
    )
    assert d.desired_instances == 90
    assert d.quota_limited is True
    assert d.reason == "clamped_quota"


# ---------------------------------------------------------------------------
# D. Anti-thrash: scale-in held until K idle ticks; scale-out ignores it.
# ---------------------------------------------------------------------------

def test_scale_in_held_until_k_idle_ticks():
    # Queue empty, currently at 20. Only 1 idle tick (< default K=3) -> hold.
    held = compute_desired_capacity(0, consecutive_idle_ticks=1, current_instances=20)
    assert held.desired_instances == 20
    assert held.reason == "scale_in_held"

    # Once K idle ticks have elapsed, drop to the floor.
    dropped = compute_desired_capacity(0, consecutive_idle_ticks=3, current_instances=20)
    assert dropped.desired_instances == 1


def test_scale_out_immediate_ignores_idle_counter():
    # Heavy backlog but the idle counter is mid-count. Scale-OUT must still fire
    # immediately — the idle hold only ever gates scale-IN.
    d = compute_desired_capacity(600, consecutive_idle_ticks=2, current_instances=1)
    assert d.desired_instances == 45
    assert d.reason == "demand"


# ---------------------------------------------------------------------------
# E. Floors and robustness.
# ---------------------------------------------------------------------------

def test_never_below_min_and_never_crashes():
    for depth in (0, -5, 1, 10, 10_000):
        d = compute_desired_capacity(depth)
        assert d.desired_instances >= 1


def test_custom_min_instances_is_a_floor():
    d = compute_desired_capacity(0, overrides={"min_instances": 3})
    assert d.desired_instances >= 3


@pytest.mark.parametrize(
    "overrides",
    [
        {"sla_seconds": 0},
        {"per_file_seconds": 0},
        {"per_instance_concurrency": 0},
        {"sla_seconds": -10, "per_file_seconds": -1, "per_instance_concurrency": -5},
        {"max_instances": 0, "min_instances": 0},
        {"scale_in_idle_ticks": 0},
    ],
)
def test_degenerate_overrides_never_crash(overrides):
    # Degenerate operator JSON (zeros / negatives) must clamp, never divide-by-zero.
    d = compute_desired_capacity(1_000, overrides=overrides)
    assert d.desired_instances >= 1
    assert d.drain_eta_seconds >= 0


# ---------------------------------------------------------------------------
# F. pool_quota_ceiling — aggregate-TPM, scales linearly, binds on smaller lane.
# ---------------------------------------------------------------------------

# Chosen so the EMBEDDING lane is the binding (smaller files/min) constraint:
#   chat:  9,000,000 tpm / 10,000 tpf = 900 files/min
#   embed:   500,000 tpm /    700 tpf ~ 714 files/min  (smaller -> binds)
_CHAT_TPM = 9_000_000
_EMBED_TPM = 500_000
_TPF_CHAT = 10_000
_TPF_EMBED = 700


def _ceiling_for(n_chat: int, n_embed: int, per_file_seconds: float = 45.0):
    deployments = [_dep("chat", _CHAT_TPM) for _ in range(n_chat)]
    deployments += [_dep("embedding", _EMBED_TPM) for _ in range(n_embed)]
    return pool_quota_ceiling(deployments, _TPF_CHAT, _TPF_EMBED, per_file_seconds)


def test_pool_ceiling_empty_is_none():
    assert pool_quota_ceiling([], _TPF_CHAT, _TPF_EMBED, 45.0) is None
    assert pool_quota_ceiling(None, _TPF_CHAT, _TPF_EMBED, 45.0) is None


def test_pool_ceiling_binds_on_smaller_lane():
    # 1 chat + 1 embed: embed (714 f/min) binds, not chat (900 f/min).
    # floor(714.28... / 60 * 45) = floor(535.7) = 535.
    c = _ceiling_for(1, 1)
    expected_embed = (_EMBED_TPM / _TPF_EMBED) / 60.0 * 45.0
    expected_chat = (_CHAT_TPM / _TPF_CHAT) / 60.0 * 45.0
    assert c == int(expected_embed)  # floor of the smaller lane
    assert c < int(expected_chat)  # chat lane would have allowed more


def test_pool_ceiling_scales_linearly_with_deployments():
    # Scale the BINDING (embedding) lane while keeping the chat lane effectively
    # unlimited (1 huge chat deployment) so embed stays the constraint. Adding
    # embed deployments must then multiply the ceiling ~linearly. (If chat were
    # finite it would cap once embed overtakes it — that crossover is exactly
    # what binds-on-smaller-lane means, and is covered separately.)
    huge_chat = [_dep("chat", _CHAT_TPM * 100)]

    def ceiling(n_embed: int):
        deployments = list(huge_chat) + [_dep("embedding", _EMBED_TPM) for _ in range(n_embed)]
        return pool_quota_ceiling(deployments, _TPF_CHAT, _TPF_EMBED, 45.0)

    one = ceiling(1)
    two = ceiling(2)
    four = ceiling(4)
    assert two == pytest.approx(one * 2, rel=0.001, abs=1)
    assert four == pytest.approx(one * 4, rel=0.001, abs=1)


def test_pool_ceiling_degenerate_tokens_never_crash():
    # Zero / negative tokens-per-file must clamp to 1, never divide-by-zero.
    c = pool_quota_ceiling([_dep("chat", _CHAT_TPM), _dep("embedding", _EMBED_TPM)], 0, -5, 45.0)
    assert c is not None and c >= 1
    # Zero per_file_seconds clamps to the 0.001 floor and still returns >= 1.
    c2 = pool_quota_ceiling([_dep("chat", _CHAT_TPM)], _TPF_CHAT, _TPF_EMBED, 0)
    assert c2 is not None and c2 >= 1


def test_pool_clamp_wins_in_compute_desired_capacity():
    # Inject a tiny pool via the _deployments override key. The aggregate-TPM
    # ceiling must clamp desired below the demand number and flag quota_limited.
    deployments = [_dep("chat", _CHAT_TPM), _dep("embedding", _EMBED_TPM)]
    ceiling = pool_quota_ceiling(deployments, _TPF_CHAT, _TPF_EMBED, 45.0)
    d = compute_desired_capacity(
        100_000,
        overrides={"_deployments": deployments, "max_instances": 5_000},
    )
    assert d.quota_limited is True
    assert d.reason == "clamped_quota"
    assert d.desired_instances == ceiling  # per_instance_concurrency defaults to 1


def test_pool_clamp_takes_precedence_over_single_rpm():
    # When BOTH a pool and ingest_rpm are present, the pool ceiling is used.
    deployments = [_dep("chat", _CHAT_TPM), _dep("embedding", _EMBED_TPM)]
    pool_ceiling = pool_quota_ceiling(deployments, _TPF_CHAT, _TPF_EMBED, 45.0)
    d = compute_desired_capacity(
        100_000,
        overrides={
            "_deployments": deployments,
            "ingest_rpm": 1,  # would give a tiny single-RPM ceiling if it won
            "max_instances": 5_000,
        },
    )
    assert d.desired_instances == pool_ceiling
