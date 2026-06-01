"""Autoscale decision plane for ingestion workers (<=10-minute drain SLA).

Mirror of ``resource_profile.py``: a *pure* formula (``compute_desired_capacity``)
that maps a measured queue depth to a desired worker count, plus a *thin impure*
collector (``read_queue_depth``) that reads Redis ``LLEN`` across the ingest
queues. Same split as the resource formula so the decision math is trivially
unit-testable: feed it a number and assert on the ``ScaleDecision``.

DECISION vs ACTUATION. This module never calls Azure to resize anything. It
publishes a NUMBER (log + metric); a dumb, auditable Azure autoscale rule (with
its own hard ceiling) actuates. Separating the decision (here, unit-tested, no
cloud SDK) from the action (a cloud rule that cannot exceed its max-count) is the
safety posture — a bug here cannot launch 500 VMs.

Two clamps bound the desired count:

1. Operator bounds ``[min_instances, max_instances]`` (budget guardrails).
2. The LLM quota ceiling — the real bottleneck. When a model pool is configured
   we use ``pool_quota_ceiling`` (min-over-lanes aggregate-TPM), which scales
   ~linearly with the number of deployments and binds on the smaller lane
   (usually embeddings). Absent a pool we fall back to the legacy single-RPM
   ceiling (``_quota_concurrency_ceiling``).

Anti-thrash: scale-OUT is immediate; scale-IN is held until ``K`` consecutive
idle ticks have elapsed (fast-out / slow-in), so we never flap a worker off and
immediately need it back.

``redis`` is a soft dependency, guarded exactly like ``psutil`` in
``resource_profile.py`` — the module imports and the pure formula runs with no
redis/openai installed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# redis is a soft dependency: the pure decision formula must import and run with
# no broker library present. Guard it like psutil in resource_profile.py.
try:  # pragma: no cover - import guard
    import redis as _redis
except Exception:  # noqa: BLE001 - any import failure means "not available"
    _redis = None  # type: ignore[assignment]


# Prefer the project's named pipeline logger; fall back to a plain structlog
# logger so this module works even if the logging module is reshaped.
try:  # pragma: no cover - import guard
    from app.core.logger import pipeline_logger as _logger
except Exception:  # noqa: BLE001
    import structlog

    _logger = structlog.get_logger("ingestion_autoscale")


# The 10-minute SLA, in seconds — the constant the whole design is built around.
_DEFAULT_SLA_SECONDS = 600


# Default tuning constants. Every one is overridable via the ``overrides`` dict
# (the policy ``resource_model`` block) so ops can tune without a code change.
# ``ingest_rpm`` is SHARED with compute_ingestion_knobs — one number caps both
# per-worker IO concurrency AND how many workers are worth running.
_DEFAULTS: dict[str, float] = {
    "sla_seconds": _DEFAULT_SLA_SECONDS,
    "per_file_seconds": 45.0,         # conservative end-to-end median; tune down
    "per_instance_concurrency": 1,    # files in flight per worker instance
    "min_instances": 1,               # warm baseline (broker + metric publisher)
    "max_instances": 50,              # operator/budget hard ceiling
    "scale_in_idle_ticks": 3,         # K consecutive idle reads before scale-in
    "ingest_rpm": 0,                  # >0 caps sustainable concurrency (legacy fallback)
    # Pool-aware aggregate-TPM clamp inputs (see pool_quota_ceiling). Used only
    # when deployments are present in the overrides.
    "tokens_per_file_chat": 10000,
    "tokens_per_file_embed": 700,
}


@dataclass(frozen=True)
class ScaleDecision:
    """Immutable result of one capacity decision.

    ``desired_instances`` is the number the autoscale rule should converge to.
    ``reason`` is a short human string (for logs/metrics) describing which clamp
    won. ``quota_limited`` is True when the LLM quota ceiling — not demand — set
    the number (the honest "buy more quota" signal). ``drain_eta_seconds`` is the
    projected time to clear the current backlog at ``desired_instances``.
    """

    desired_instances: int
    reason: str
    quota_limited: bool
    drain_eta_seconds: int


# ---------------------------------------------------------------------------
# Pure quota math — no I/O. Feed it numbers / tiny deployment objects.
# ---------------------------------------------------------------------------

def _quota_concurrency_ceiling(ingest_rpm: int, per_file_seconds: float) -> int | None:
    """Legacy single-deployment RPM ceiling: max sustainable in-flight files.

    Kept as the graceful-degradation floor for when no model pool is configured.
    ``None`` disables the clamp (ingest_rpm <= 0).
    """
    if ingest_rpm <= 0:
        return None
    per_file_seconds = max(0.001, float(per_file_seconds))
    sustainable_rps = ingest_rpm / 60.0
    return max(1, math.floor(sustainable_rps * per_file_seconds))


def pool_quota_ceiling(
    deployments,
    tokens_per_file_chat,
    tokens_per_file_embed,
    per_file_seconds,
) -> int | None:
    """PURE: max sustainable IN-FLIGHT files bounded by aggregate pool TPM.

        pool_files_per_min = min_over_lanes( sum(lane.tpm) / tokens_per_file_lane )
        quota_max_inflight = floor(pool_files_per_min / 60 * per_file_seconds)

    Scales ~linearly with the number of deployments and binds on the SMALLER
    lane (usually embeddings). ``None`` when there are no deployments (the caller
    then keeps the legacy single-RPM clamp). Every input is clamped so a stray
    0/negative cannot crash or divide-by-zero.

    ``deployments`` is any iterable of objects exposing ``.kind`` ("chat" /
    "embedding") and ``.tpm`` (tokens-per-minute), e.g. ``model_pool.Deployment``.
    """
    if not deployments:
        return None
    per_file_seconds = max(0.001, float(per_file_seconds))
    chat_tpm = sum(
        max(0, int(getattr(d, "tpm", 0))) for d in deployments if getattr(d, "kind", "") == "chat"
    )
    embed_tpm = sum(
        max(0, int(getattr(d, "tpm", 0))) for d in deployments if getattr(d, "kind", "") == "embedding"
    )
    tpf_chat = max(1, int(tokens_per_file_chat))
    tpf_embed = max(1, int(tokens_per_file_embed))
    lanes = []
    if chat_tpm > 0:
        lanes.append(chat_tpm / tpf_chat)
    if embed_tpm > 0:
        lanes.append(embed_tpm / tpf_embed)
    if not lanes:
        return None
    return max(1, math.floor(min(lanes) / 60.0 * per_file_seconds))


def _extract_deployments(overrides: dict | None):
    """Best-effort: pull tiny TPM-bearing deployment objects from overrides.

    Accepts either an already-built list under ``_deployments`` (the form the
    control loop injects) or a ``model_pool``/``deployments`` config block which
    we parse via ``app.core.model_pool.load_deployments`` when that module
    exists. Returns an empty tuple when no pool is configured so the caller falls
    back to the single-RPM clamp. Never raises.
    """
    if not overrides:
        return ()
    pre = overrides.get("_deployments")
    if pre:
        return tuple(pre)
    raw = None
    pool = overrides.get("model_pool")
    if isinstance(pool, dict):
        raw = pool.get("deployments")
    if raw is None:
        raw = overrides.get("deployments")
    if not raw:
        return ()
    try:  # pragma: no cover - exercised only when model_pool exists
        from app.core.model_pool import load_deployments

        return tuple(load_deployments(raw))
    except Exception:  # noqa: BLE001
        return ()


# ---------------------------------------------------------------------------
# Pure formula — desired worker count from a measured queue depth. No I/O.
# ---------------------------------------------------------------------------

def compute_desired_capacity(
    queue_depth: int,
    overrides: dict | None = None,
    *,
    consecutive_idle_ticks: int = 0,
    current_instances: int | None = None,
) -> ScaleDecision:
    """Pure mapping from a measured queue depth to a desired worker count.

        desired = ceil(queue_depth * per_file_seconds
                       / (sla_seconds * per_instance_concurrency))

    clamped to ``[min_instances, max_instances]`` and by the LLM quota ceiling
    (pool aggregate-TPM when deployments are present, else the single-RPM
    fallback). Scale-OUT is immediate; scale-IN is held until ``K`` consecutive
    idle reads (anti-thrash). Every operator input is clamped so a stray
    0/negative cannot crash or divide-by-zero.
    """
    o = dict(_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in o:
                o[key] = value

    # Clamp EVERY input — a stray 0/negative from operator JSON must never crash.
    queue_depth = max(0, int(queue_depth))
    sla_seconds = max(1, int(o["sla_seconds"]))
    per_file_seconds = max(0.001, float(o["per_file_seconds"]))
    per_instance_concurrency = max(1, int(o["per_instance_concurrency"]))
    min_instances = max(1, int(o["min_instances"]))
    max_instances = max(min_instances, int(o["max_instances"]))
    scale_in_idle_ticks = max(1, int(o["scale_in_idle_ticks"]))
    ingest_rpm = int(o["ingest_rpm"]) if o["ingest_rpm"] else 0

    # Step 1 — SLA-driven demand.
    raw_desired = math.ceil(
        (queue_depth * per_file_seconds) / (sla_seconds * per_instance_concurrency)
    )

    # Step 2 — clamp by operator bounds.
    desired = max(min_instances, min(max_instances, raw_desired))
    reason = "demand"
    if raw_desired > max_instances:
        reason = "clamped_max"
    elif raw_desired < min_instances:
        reason = "clamped_min"

    # Step 3 — clamp by the hard quota ceiling. Prefer the pool aggregate-TPM
    # ceiling (scales linearly with #deployments); fall back to single-RPM.
    quota_limited = False
    deployments = _extract_deployments(overrides)
    ceiling = pool_quota_ceiling(
        deployments,
        int(o["tokens_per_file_chat"]),
        int(o["tokens_per_file_embed"]),
        per_file_seconds,
    )
    if ceiling is None:
        ceiling = _quota_concurrency_ceiling(ingest_rpm, per_file_seconds)
    if ceiling is not None:
        quota_instances = max(min_instances, ceiling // per_instance_concurrency)
        if quota_instances < desired:
            desired = quota_instances
            quota_limited = True
            reason = "clamped_quota"

    # Step 4 — anti-thrash: scale-IN held until K consecutive idle ticks.
    # Scale-OUT is never held (a higher desired always wins immediately).
    if current_instances is not None:
        current = max(0, int(current_instances))
        if desired < current and consecutive_idle_ticks < scale_in_idle_ticks:
            desired = max(min_instances, current)
            reason = "scale_in_held"

    # Honest ETA at the chosen capacity (>=1 effective concurrency by clamps).
    effective_concurrency = max(1, desired * per_instance_concurrency)
    drain_eta_seconds = math.ceil(queue_depth * per_file_seconds / effective_concurrency)

    return ScaleDecision(int(desired), reason, quota_limited, int(drain_eta_seconds))


# ---------------------------------------------------------------------------
# Thin impure collector — Redis IO only (split like resource_profile.py).
# ---------------------------------------------------------------------------

def read_queue_depth(queue_names: tuple[str, ...] | None = None) -> int:
    """Total pending files across the Redis ingest queues (``LLEN``).

    Returns 0 on any failure (redis absent, broker unreachable, bad key) so the
    decision loop degrades to "no backlog observed" rather than crashing. The
    queue names default to the real ingest queues (settings-driven, with the
    hardcoded ``ingest_high/normal/low`` literals as the floor).
    """
    if _redis is None:
        return 0
    try:
        from app.core.config import get_settings

        settings = get_settings()
        names = queue_names or (
            getattr(settings, "INGEST_HIGH_QUEUE", "ingest_high"),
            getattr(settings, "INGEST_NORMAL_QUEUE", "ingest_normal"),
            getattr(settings, "INGEST_LOW_QUEUE", "ingest_low"),
        )
        client = _redis.Redis.from_url(settings.REDIS_URL)
        total = 0
        for name in names:
            try:
                total += int(client.llen(name))
            except Exception:  # noqa: BLE001 - skip an unreadable queue, keep the rest
                continue
        return max(0, total)
    except Exception:  # noqa: BLE001
        return 0


def scaler_tick(
    *,
    consecutive_idle_ticks: int = 0,
    current_instances: int | None = None,
    overrides: dict | None = None,
) -> ScaleDecision:
    """One observe -> decide -> emit cycle. Emits a number; never resizes.

    Reuses the SAME override source the resource formula uses
    (``ingestion_policy._resource_model_overrides``) so the autoscaler honours
    the policy ``resource_model`` block with ENV > JSON > default precedence. The
    caller (a sidecar loop or a Celery-beat task) owns ``consecutive_idle_ticks``
    and ``current_instances``.
    """
    if overrides is None:
        try:
            from app.services.ingestion_policy import _resource_model_overrides

            overrides = _resource_model_overrides()
        except Exception:  # noqa: BLE001
            overrides = None

    queue_depth = read_queue_depth()
    decision = compute_desired_capacity(
        queue_depth,
        overrides,
        consecutive_idle_ticks=consecutive_idle_ticks,
        current_instances=current_instances,
    )

    try:
        _logger.info(
            "ingestion_autoscale.tick",
            queue_depth=queue_depth,
            desired_instances=decision.desired_instances,
            reason=decision.reason,
            quota_limited=decision.quota_limited,
            drain_eta_seconds=decision.drain_eta_seconds,
            consecutive_idle_ticks=consecutive_idle_ticks,
            current_instances=current_instances,
        )
    except Exception:  # noqa: BLE001 - logging must never break the decision
        pass

    # Emit gauges via the in-process metrics module. The real metrics API has no
    # gauge() (only inc/dec on a fixed counter dict), so set the latest value
    # directly under its lock. Guarded so a metrics reshape can never break the
    # tick — the decision is what matters, not the gauge.
    try:
        from app.core import metrics

        with metrics._lock:  # type: ignore[attr-defined]
            metrics._counters["ingest_queue_depth"] = queue_depth  # type: ignore[attr-defined]
            metrics._counters["ingest_desired_instances"] = decision.desired_instances  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        pass

    return decision
