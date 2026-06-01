"""Multi-deployment model pool: pure weighted selection + a thin failover shell.

This module is the single chokepoint for "given a fleet of Azure OpenAI
deployments (lanes), which one should this request hit, and what happens when a
lane returns 429/timeout?". It mirrors the design split that
``app/services/resource_profile.py`` established:

1. ``load_deployments()`` / ``select_deployment()`` — *pure* functions. They
   parse operator config into immutable ``Deployment`` records and perform a
   deterministic, weighted, health-aware lane pick. No I/O, no globals, no
   network — feed them hand-built deployments + a health map in a unit test and
   assert on the chosen lane.

2. ``ModelPool`` — a *thin async IO shell*. It owns lazily-constructed
   per-endpoint ``AsyncAzureOpenAI`` clients, a per-deployment circuit breaker
   (a lane that raises ``RateLimitError``/``APITimeoutError`` is tripped into a
   cooldown), and a bounded failover loop on top of ``select_deployment``. All
   the decision logic lives in the pure layer; the shell only does the call.

Graceful degradation is the floor: absent/garbage config collapses to the
single *legacy* deployment built from today's settings, so the pool behaves
exactly like the current ``get_client()`` single-deployment path. ``openai`` is
a soft dependency — importing this module never requires it; only an actual
network ``_call`` does.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

# openai is a soft dependency. The pure selection/parse layer must import and be
# unit-testable without it; only an actual network call needs the real client.
try:  # pragma: no cover - import guard
    from openai import APITimeoutError, AsyncAzureOpenAI, RateLimitError
except Exception:  # noqa: BLE001 - any import failure means "not available"
    AsyncAzureOpenAI = None  # type: ignore[assignment,misc]
    RateLimitError = APITimeoutError = Exception  # type: ignore[assignment,misc]


# Prefer the project's named pipeline logger; fall back to a plain structlog
# logger so this module works even if the logging module is reshaped.
try:  # pragma: no cover - import guard
    from app.core.logger import pipeline_logger as _logger
except Exception:  # noqa: BLE001
    import structlog

    _logger = structlog.get_logger("model_pool")


@dataclass(frozen=True)
class Deployment:
    """Immutable description of one Azure OpenAI deployment ("lane").

    ``kind`` partitions the fleet into "chat" and "embedding" lanes (selection
    never crosses kinds). ``weight`` biases the weighted pick within a kind.
    ``tier`` is "standard" or "high" — a ``tier='high'`` request prefers high
    lanes (e.g. gpt-4o) and falls back to standard only if none exist.
    """

    name: str
    kind: str
    endpoint: str
    deployment_id: str
    rpm: int
    tpm: int
    weight: float
    region: str
    tier: str = "standard"


@dataclass(frozen=True)
class HealthState:
    """Per-deployment circuit-breaker state.

    ``cooling_until`` is a ``time.monotonic()`` deadline: ``0.0`` (or any value
    <= now) means healthy; a value greater than now means the lane is in
    cooldown after tripping the breaker.
    """

    cooling_until: float = 0.0


# Default control-plane tuning. Every value is overridable via the ``overrides``
# dict (typically the policy ``model_pool`` block) so ops can tune without a
# code change. None/missing in overrides falls back to these.
_DEFAULTS: dict[str, float] = {
    "circuit_breaker_cooldown_seconds": 20.0,
    "request_timeout_seconds": 30.0,
    "max_failover_attempts": 4,
    "escalation_confidence_threshold": 0.55,
}


# ---------------------------------------------------------------------------
# Pure parsing + selection — no I/O, no globals. Feed them data in a test.
# ---------------------------------------------------------------------------


def load_deployments(
    raw: object,
    *,
    legacy: Deployment | None = None,
) -> tuple[Deployment, ...]:
    """Pure: parse JSON-string / list-of-dicts config into Deployments.

    Accepts either a JSON string (an array of objects), or an already-parsed
    list/tuple of dicts (as ingestion policy would hand it over). Every numeric
    field is clamped so a stray 0/negative can't yield a degenerate lane, and
    rows of an unknown ``kind`` are skipped.

    Empty or garbage config never boots an empty pool: it degrades to the single
    ``legacy`` deployment (today's behaviour) when one is supplied, else an empty
    tuple (the caller then has no lanes — ``ModelPool._call`` will raise clearly).
    """
    items: list[dict] = []
    try:
        if isinstance(raw, str) and raw.strip():
            parsed = json.loads(raw)
            items = [d for d in parsed if isinstance(d, dict)] if isinstance(parsed, list) else []
        elif isinstance(raw, (list, tuple)):
            items = [d for d in raw if isinstance(d, dict)]
    except (ValueError, TypeError):
        items = []

    out: list[Deployment] = []
    for d in items:
        try:
            kind = str(d.get("kind", "chat")).strip().lower()
            if kind not in ("chat", "embedding"):
                continue
            out.append(
                Deployment(
                    name=str(d.get("name") or d.get("deployment_id") or "unnamed"),
                    kind=kind,
                    endpoint=str(d.get("endpoint", "")),
                    deployment_id=str(d.get("deployment_id", "")),
                    rpm=max(1, int(d.get("rpm", 1))),
                    tpm=max(1, int(d.get("tpm", 1))),
                    weight=max(0.0, float(d.get("weight", 1.0))),
                    region=str(d.get("region", "")),
                    tier=("high" if str(d.get("tier", "standard")).strip().lower() == "high" else "standard"),
                )
            )
        except (TypeError, ValueError):
            continue

    if not out:
        return (legacy,) if legacy is not None else tuple()
    return tuple(out)


def legacy_deployment_from_settings(settings: object, *, kind: str = "chat") -> Deployment:
    """Build the single legacy ``Deployment`` from real settings.

    This is the graceful-degradation floor: when no ``model_pool`` config exists,
    a pool seeded with just this lane behaves exactly like today's single
    ``get_client()`` path. Attribute names mirror the real ``Settings`` used by
    ``openai_client.py`` / ``retrieval/embeddings.py``:
    ``AZURE_OPENAI_ENDPOINT`` (with ``AZURE_OPENAI_API_BASE`` fallback) and the
    per-kind deployment name. No network — pure read of settings attributes.
    """

    def _attr(name: str, default: str = "") -> str:
        return str(getattr(settings, name, default) or default)

    endpoint = _attr("AZURE_OPENAI_ENDPOINT") or _attr("AZURE_OPENAI_API_BASE")
    if kind == "embedding":
        deployment_id = _attr("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
        name = deployment_id or "legacy-embedding"
    else:
        deployment_id = _attr("AZURE_OPENAI_DEPLOYMENT")
        name = deployment_id or "legacy-chat"

    return Deployment(
        name=name,
        kind=kind,
        endpoint=endpoint,
        deployment_id=deployment_id,
        rpm=1,
        tpm=1,
        weight=1.0,
        region="",
        tier="standard",
    )


def select_deployment(
    deployments: tuple[Deployment, ...],
    health: dict[str, HealthState],
    rng_token: float,
    *,
    kind: str = "chat",
    tier: str = "standard",
    now: float | None = None,
) -> Deployment | None:
    """Pure, deterministic weighted lane pick across HEALTHY lanes of ``kind``.

    Rules (in order):
      * Only lanes whose ``kind`` matches are candidates; ``None`` if none.
      * ``tier='high'`` narrows to high lanes when any exist, else falls back to
        the full candidate set (degrade, never fail).
      * A lane whose ``cooling_until > now`` is skipped. If EVERY candidate is
        cooling, return the one that recovers soonest (smallest ``cooling_until``).
      * Among healthy lanes, pick by clamped weight: ``rng_token`` in [0,1) maps
        onto the cumulative-weight line, so the same token yields the same lane
        (deterministic). All-zero weights degrade to a uniform pick by token.

    No I/O, no globals — ``now`` and ``rng_token`` are injected so tests are
    fully deterministic.
    """
    now = time.monotonic() if now is None else now

    candidates = [d for d in deployments if d.kind == kind]
    if not candidates:
        return None

    if tier == "high":
        high = [d for d in candidates if d.tier == "high"]
        if high:
            candidates = high

    def _cooling_until(d: Deployment) -> float:
        return health.get(d.name, HealthState()).cooling_until

    healthy = [d for d in candidates if _cooling_until(d) <= now]
    if not healthy:
        # Everything is cooling — pick the soonest-to-recover lane.
        return min(candidates, key=_cooling_until)

    token = min(1.0, max(0.0, float(rng_token)))
    total = sum(d.weight for d in healthy)
    if total <= 0.0:
        # All-zero weights: uniform pick keyed off the token (deterministic).
        return healthy[min(len(healthy) - 1, int(token * len(healthy)))]

    target, cumulative = token * total, 0.0
    for d in healthy:
        cumulative += d.weight
        if target < cumulative:
            return d
    return healthy[-1]


# ---------------------------------------------------------------------------
# Thin async IO shell — owns clients + breaker + failover. Inject everything.
# ---------------------------------------------------------------------------


class ModelPool:
    """Thin async shell over ``select_deployment``: weighted pick + failover.

    Constructed from parsed ``deployments`` and an optional ``overrides`` dict
    (the policy ``model_pool`` block). Holds NO module globals — health and
    clients live on the instance. ``rng`` is injectable so failover order is
    deterministic in tests; it defaults to ``random.random``.

    A lane that raises ``RateLimitError`` / ``APITimeoutError`` trips its circuit
    breaker (cooldown) and the call fails over to the next lane, up to
    ``max_failover_attempts`` distinct lanes.
    """

    def __init__(
        self,
        deployments: tuple[Deployment, ...],
        overrides: dict | None = None,
        *,
        rng=None,
    ) -> None:
        o = dict(_DEFAULTS)
        if overrides:
            for key, value in overrides.items():
                if value is not None and key in o:
                    o[key] = value

        self._deployments: tuple[Deployment, ...] = tuple(deployments)
        # Clamp every operator-supplied control knob to a safe range.
        self._cooldown = max(0.0, float(o["circuit_breaker_cooldown_seconds"]))
        self._timeout = max(0.001, float(o["request_timeout_seconds"]))
        self._max_attempts = max(1, int(o["max_failover_attempts"]))
        self._health: dict[str, HealthState] = {}
        self._clients: dict[str, object] = {}

        import random

        self._rng = rng or random.random

    def _client_for(self, d: Deployment):
        """Lazily build (and cache) the AsyncAzureOpenAI client for a lane.

        Cached per ``endpoint|deployment_id`` so co-located lanes share a client.
        Endpoint comes from the lane, falling back to the legacy single-endpoint
        settings; key/api_version always come from settings.
        """
        if AsyncAzureOpenAI is None:
            raise RuntimeError("openai not installed; ModelPool cannot make network calls")
        key = f"{d.endpoint}|{d.deployment_id}"
        client = self._clients.get(key)
        if client is None:
            from app.core.config import get_settings

            s = get_settings()
            client = AsyncAzureOpenAI(
                azure_endpoint=d.endpoint or s.AZURE_OPENAI_ENDPOINT or s.AZURE_OPENAI_API_BASE,
                api_key=s.AZURE_OPENAI_KEY or s.AZURE_OPENAI_API_KEY,
                api_version=s.AZURE_OPENAI_API_VERSION,
                timeout=self._timeout,
            )
            self._clients[key] = client
        return client

    def _trip(self, d: Deployment) -> None:
        """Open the circuit breaker for a lane (start its cooldown window)."""
        self._health[d.name] = HealthState(cooling_until=time.monotonic() + self._cooldown)
        try:
            _logger.warning(
                "model_pool.cooling",
                deployment=d.name,
                kind=d.kind,
                region=d.region,
                cooldown_s=self._cooldown,
            )
        except Exception:  # noqa: BLE001 - logging must never break the call path
            pass

    async def acomplete(self, *, messages, tier: str = "standard", **kw):
        """Chat completion with weighted lane selection + 429/timeout failover."""
        return await self._call("chat", tier, messages=messages, **kw)

    async def aembed(self, *, inputs, tier: str = "standard", **kw):
        """Embedding request with weighted lane selection + 429/timeout failover."""
        return await self._call("embedding", tier, inputs=inputs, **kw)

    async def _call(self, kind: str, tier: str, **payload):
        tried: set[str] = set()
        last_exc: Exception | None = None
        for _ in range(self._max_attempts):
            d = select_deployment(
                self._deployments,
                self._health,
                self._rng(),
                kind=kind,
                tier=tier,
            )
            if d is None or d.name in tried:
                break
            tried.add(d.name)
            try:
                client = self._client_for(d)
                if kind == "chat":
                    return await client.chat.completions.create(
                        model=d.deployment_id,
                        messages=payload["messages"],
                        **{k: v for k, v in payload.items() if k != "messages"},
                    )
                return await client.embeddings.create(
                    model=d.deployment_id,
                    input=payload["inputs"],
                    **{k: v for k, v in payload.items() if k != "inputs"},
                )
            except (RateLimitError, APITimeoutError) as exc:
                last_exc = exc
                self._trip(d)
                continue
        raise last_exc or RuntimeError(f"model_pool: no healthy {kind} deployment")
