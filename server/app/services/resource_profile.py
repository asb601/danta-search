"""VM/container resource detection + a pure ingestion-knob formula.

This module is the single source of truth for "how big is the box we are
running on, and how many concurrent ingestion jobs can it safely sustain?".

Two responsibilities, deliberately split so the second is trivially testable:

1. ``get_resource_profile()`` — *impure* detection. Reads cgroup files,
   ``/proc``-style sysconf values, and (optionally) ``psutil``. It is
   cgroup-aware on purpose: inside a container the host may advertise 64 cores
   and 256 GiB while the cgroup only grants this process 2 cores and 4 GiB. If
   we sized pools off the host we would oversubscribe and OOM/throttle. The
   detection therefore always takes the *minimum* of every signal that bounds
   us. Result is ``lru_cache``d — the box does not change shape at runtime.

2. ``compute_ingestion_knobs()`` — *pure* function. Given a ``ResourceProfile``
   (and optional override constants), it returns the concrete concurrency /
   memory knobs the ingestion layer should use. No I/O, no globals — feed it a
   hand-built profile in a unit test and assert on the dict.

``psutil`` is an optional accelerator, not a hard dependency. Everything
degrades gracefully (``psutil = None``) so the app still boots without it.
"""

from __future__ import annotations

import math
import os
import shutil
import tempfile
from dataclasses import dataclass
from functools import lru_cache

# psutil is a soft dependency. It gives us nicer host RAM/CPU signals, but the
# cgroup + sysconf paths cover the cases we actually care about, so guard it.
try:  # pragma: no cover - import guard
    import psutil
except Exception:  # noqa: BLE001 - any import failure means "not available"
    psutil = None  # type: ignore[assignment]


# Prefer the project's named pipeline logger; fall back to a plain structlog
# logger so this module works even if the logging module is reshaped.
try:  # pragma: no cover - import guard
    from app.core.logger import pipeline_logger as _logger
except Exception:  # noqa: BLE001
    import structlog

    _logger = structlog.get_logger("resource_profile")


# cgroup v1's "unlimited" memory limit is reported as a huge sentinel
# (PAGE_SIZE * 2**63-ish). Anything at/above this magnitude is "no limit".
_CGROUP_V1_UNLIMITED_THRESHOLD = 1 << 62


@dataclass(frozen=True)
class ResourceProfile:
    """Immutable snapshot of the resources this process is actually allowed.

    All byte fields are absolute bytes. ``source`` is a short human string
    describing which signal won (for logs/debugging). ``is_container`` is True
    when a cgroup limit was the binding constraint for CPU or RAM.
    """

    cpu_count: int
    ram_total_bytes: int
    ram_available_bytes: int
    disk_free_bytes: int
    source: str
    is_container: bool


# Module-level flag so the resolved profile is logged exactly once per process.
_PROFILE_LOGGED = False


def _read_int_file(path: str) -> int | None:
    """Read a single integer from a cgroup file; None if absent/unreadable."""
    try:
        with open(path, encoding="utf-8") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _cgroup_cpu_quota() -> int | None:
    """CPU core cap imposed by a cgroup, rounded DOWN, or None if uncapped.

    cgroup v2: ``/sys/fs/cgroup/cpu.max`` -> "<quota> <period>" or "max <period>".
    cgroup v1: ``cpu.cfs_quota_us`` / ``cpu.cfs_period_us`` (quota -1 == none).

    For CPU-bound pool sizing we round a fractional quota DOWN (a 1.5-core
    quota means we should plan for 1 dedicated core, never 2).
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as handle:
            raw = handle.read().strip().split()
        if raw and raw[0] != "max":
            quota = int(raw[0])
            period = int(raw[1]) if len(raw) > 1 else 100_000
            if quota > 0 and period > 0:
                return max(1, math.floor(quota / period))
    except (OSError, ValueError, IndexError):
        pass

    # cgroup v1
    quota = _read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    period = _read_int_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    if quota is not None and quota > 0 and period and period > 0:
        return max(1, math.floor(quota / period))

    return None


def _cgroup_memory_limit() -> int | None:
    """RAM ceiling imposed by a cgroup, or None if uncapped/unreadable.

    cgroup v2: ``/sys/fs/cgroup/memory.max`` ("max" == unlimited).
    cgroup v1: ``memory.limit_in_bytes`` (huge sentinel == unlimited).
    """
    # cgroup v2
    try:
        with open("/sys/fs/cgroup/memory.max", encoding="utf-8") as handle:
            raw = handle.read().strip()
        if raw and raw != "max":
            value = int(raw)
            if value > 0:
                return value
    except (OSError, ValueError):
        pass

    # cgroup v1
    value = _read_int_file("/sys/fs/cgroup/memory/memory.limit_in_bytes")
    if value is not None and 0 < value < _CGROUP_V1_UNLIMITED_THRESHOLD:
        return value

    return None


def _cgroup_memory_usage() -> int | None:
    """Current cgroup memory usage in bytes, or None if unavailable."""
    value = _read_int_file("/sys/fs/cgroup/memory.current")  # v2
    if value is not None and value >= 0:
        return value
    value = _read_int_file("/sys/fs/cgroup/memory/memory.usage_in_bytes")  # v1
    if value is not None and value >= 0:
        return value
    return None


def _sysconf_total_ram() -> int | None:
    """Host physical RAM via sysconf (SC_PAGE_SIZE * SC_PHYS_PAGES)."""
    if not hasattr(os, "sysconf"):
        return None
    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        page_count = os.sysconf("SC_PHYS_PAGES")
    except (OSError, ValueError):
        return None
    if isinstance(page_size, int) and isinstance(page_count, int) and page_size > 0 and page_count > 0:
        return page_size * page_count
    return None


def _detect_cpu_count() -> tuple[int, bool]:
    """Resolve the usable CPU core count and whether a cgroup bound it.

    We collect every signal that can *cap* us and take the minimum, so a cap is
    never exceeded:
      - cgroup CPU quota (rounded down)
      - sched affinity mask (cores this process may actually run on)
      - psutil logical CPU count
      - os.cpu_count()
    """
    signals: list[int] = []

    cgroup_cpu = _cgroup_cpu_quota()
    if cgroup_cpu is not None:
        signals.append(cgroup_cpu)

    # sched_getaffinity reflects cpuset pinning — Linux only.
    try:
        affinity = len(os.sched_getaffinity(0))  # type: ignore[attr-defined]
        if affinity > 0:
            signals.append(affinity)
    except (AttributeError, OSError):
        pass

    if psutil is not None:
        try:
            logical = psutil.cpu_count(logical=True)
            if logical:
                signals.append(int(logical))
        except Exception:  # noqa: BLE001
            pass

    signals.append(os.cpu_count() or 1)

    cpu_count = max(1, min(signals))
    is_container = cgroup_cpu is not None and cgroup_cpu <= cpu_count
    return cpu_count, is_container


def _detect_ram(cpu_is_container: bool) -> tuple[int, int, str, bool]:
    """Resolve (ram_total, ram_available, source, ram_is_container).

    ram_total takes the MIN of the cgroup limit and the host total so that a
    container's real ceiling wins. ram_available subtracts cgroup usage when
    known, else uses psutil's available figure, else an 0.85*total fallback.
    """
    cgroup_limit = _cgroup_memory_limit()
    host_total = None
    if psutil is not None:
        try:
            host_total = int(psutil.virtual_memory().total)
        except Exception:  # noqa: BLE001
            host_total = None
    if host_total is None:
        host_total = _sysconf_total_ram()

    candidates = [v for v in (cgroup_limit, host_total) if v and v > 0]
    if candidates:
        ram_total = max(1, min(candidates))
    else:
        ram_total = 1

    ram_is_container = cgroup_limit is not None and cgroup_limit <= ram_total
    source = "cgroup" if ram_is_container else ("psutil" if (psutil and host_total) else "sysconf")

    # Available RAM: prefer cgroup headroom, then psutil, then a safe fraction.
    ram_available: int | None = None
    if ram_is_container:
        usage = _cgroup_memory_usage()
        if usage is not None:
            ram_available = max(0, ram_total - usage)
    if ram_available is None and psutil is not None:
        try:
            ram_available = int(psutil.virtual_memory().available)
        except Exception:  # noqa: BLE001
            ram_available = None
    if ram_available is None:
        ram_available = int(0.85 * ram_total)

    # Never report more available than the (possibly cgroup-capped) total.
    ram_available = max(0, min(ram_available, ram_total))
    return ram_total, ram_available, source, ram_is_container


@lru_cache(maxsize=1)
def get_resource_profile() -> ResourceProfile:
    """Detect this process's true resource envelope (cgroup-aware, cached).

    Cached because the box does not change shape at runtime. The resolved
    profile is logged exactly once per process via the pipeline logger.
    """
    global _PROFILE_LOGGED

    cpu_count, cpu_is_container = _detect_cpu_count()
    ram_total, ram_available, ram_source, ram_is_container = _detect_ram(cpu_is_container)

    try:
        disk_free = shutil.disk_usage(tempfile.gettempdir()).free
    except OSError:
        disk_free = 0

    is_container = cpu_is_container or ram_is_container
    cpu_source = "cgroup" if cpu_is_container else ("psutil/affinity" if psutil else "os")
    source = f"cpu={cpu_source};ram={ram_source}"

    profile = ResourceProfile(
        cpu_count=cpu_count,
        ram_total_bytes=ram_total,
        ram_available_bytes=ram_available,
        disk_free_bytes=disk_free,
        source=source,
        is_container=is_container,
    )

    if not _PROFILE_LOGGED:
        _PROFILE_LOGGED = True
        try:
            _logger.info(
                "resource_profile.detected",
                cpu_count=profile.cpu_count,
                ram_total_gib=round(profile.ram_total_bytes / (1024 ** 3), 2),
                ram_available_gib=round(profile.ram_available_bytes / (1024 ** 3), 2),
                disk_free_gib=round(profile.disk_free_bytes / (1024 ** 3), 2),
                source=profile.source,
                is_container=profile.is_container,
                psutil_present=psutil is not None,
            )
        except Exception:  # noqa: BLE001 - logging must never break detection
            pass

    return profile


# ---------------------------------------------------------------------------
# Pure formula — sizing knobs from a profile. No I/O; feed it a profile.
# ---------------------------------------------------------------------------

# Default tuning constants. Every one is overridable via the ``overrides`` dict
# (and thus via the policy JSON ``resource_model`` block) so ops can tune
# without a code change.
_DEFAULTS: dict[str, float] = {
    "reserve_cores": None,  # None => auto (2 if cpu>2 else 1); explicit int (incl. 0) overrides
    "ram_safety": 0.60,
    "k_pandas": 10,
    "max_file_bytes": 64 * 1024 * 1024,
    "read_block": 64 * 1024 * 1024,
    "io_fanout": 8,
    "io_cap": 64,
    "per_proc_overhead": 250 * 1024 * 1024,
    # Optional Azure throughput hint: requests-per-minute budget. >0 caps
    # io_concurrency using a 3s/call assumption. None/0 disables the cap.
    "ingest_rpm": 0,
}


def compute_ingestion_knobs(
    profile: ResourceProfile,
    overrides: dict | None = None,
) -> dict[str, int]:
    """Pure mapping from a ``ResourceProfile`` to integer ingestion knobs.

    Every value is a concrete, ready-to-use integer. Concurrency knobs are
    floored to >= 1 so we never produce a zero-sized pool. The formula is
    memory-first: it asks "how many of this job's working sets fit in the safe
    fraction of available RAM?" and clamps that by the CPU budget.

    ``overrides`` may carry any of the keys in ``_DEFAULTS`` (typically the
    policy ``resource_model`` block). ``None``/missing falls back to defaults.
    """
    o = dict(_DEFAULTS)
    if overrides:
        for key, value in overrides.items():
            if value is not None and key in o:
                o[key] = value

    cpu = max(1, int(profile.cpu_count))

    # reserve_cores: None => auto (2 if cpu>2 else 1). An explicit int (including 0,
    # "reserve nothing") overrides, clamped to [0, cpu-1] so ingestion_cores stays
    # >= 1 and a negative/oversized value can never oversubscribe the CPU.
    if o["reserve_cores"] is None:
        reserve_cores = 2 if cpu > 2 else 1
    else:
        reserve_cores = max(0, min(int(o["reserve_cores"]), cpu - 1))

    # All remaining tunables come from operator-supplied JSON — clamp to safe
    # ranges so a stray 0/negative can't yield a degenerate or zero-division profile.
    ram_safety = min(1.0, max(0.01, float(o["ram_safety"])))
    k_pandas = max(1, int(o["k_pandas"]))
    max_file_bytes = max(1, int(o["max_file_bytes"]))
    read_block = max(1, int(o["read_block"]))
    io_fanout = max(1, int(o["io_fanout"]))
    io_cap = max(1, int(o["io_cap"]))
    per_proc_overhead = max(0, int(o["per_proc_overhead"]))
    ingest_rpm = int(o["ingest_rpm"]) if o["ingest_rpm"] else 0

    # CPU budget for ingestion (chat/API keeps the reserved cores).
    ingestion_cores = max(1, cpu - reserve_cores)

    # Safe slice of available RAM, and per-job working-set estimates.
    r_usable = profile.ram_available_bytes * ram_safety
    per_clean_job_ram = per_proc_overhead + k_pandas * max_file_bytes
    per_parquet_job_ram = per_proc_overhead + 2 * read_block

    # Memory-bounded concurrency, clamped by the CPU budget.
    preprocess_concurrency = max(1, min(ingestion_cores, math.floor(r_usable / per_clean_job_ram)))
    excel_preprocess_concurrency = max(1, preprocess_concurrency // 2)
    parquet_conversion_concurrency = max(
        1, min(ingestion_cores, math.floor((r_usable * 0.5) / per_parquet_job_ram))
    )
    celery_worker_concurrency = max(
        1, min(ingestion_cores, math.floor(r_usable / per_clean_job_ram), 32)
    )

    # I/O concurrency: CPU-driven fan-out, optionally capped by an RPM budget.
    io_concurrency = min(io_cap, io_fanout * cpu)
    if ingest_rpm > 0:
        # RPM -> in-flight concurrency assuming ~3s per call.
        io_concurrency = min(io_concurrency, max(1, math.floor(ingest_rpm / 60 * 3.0)))

    celery_prefetch_multiplier = 1
    reingest_batch_size = max(25, 25 * ingestion_cores)
    reingest_batch_delay_seconds = max(2, round(10 / ingestion_cores))

    # DuckDB threads are shared across the parquet pool; memory limit is the
    # safe RAM half split across that same pool.
    duckdb_threads = max(1, ingestion_cores // parquet_conversion_concurrency)
    duckdb_memory_limit_bytes = max(
        256 * 1024 * 1024,
        math.floor((r_usable * 0.5) / parquet_conversion_concurrency),
    )

    # Size router trip point: the pandas-OOM guard. A file above this is too big
    # to safely fan out at the current preprocess concurrency.
    small_file_threshold_bytes = math.floor(
        r_usable / (k_pandas * preprocess_concurrency * 1.5)
    )

    return {
        "ingestion_cores": ingestion_cores,
        "preprocess_concurrency": preprocess_concurrency,
        "excel_preprocess_concurrency": excel_preprocess_concurrency,
        "parquet_conversion_concurrency": parquet_conversion_concurrency,
        "celery_worker_concurrency": celery_worker_concurrency,
        "io_concurrency": io_concurrency,
        "celery_prefetch_multiplier": celery_prefetch_multiplier,
        "reingest_batch_size": reingest_batch_size,
        "reingest_batch_delay_seconds": reingest_batch_delay_seconds,
        "duckdb_threads": duckdb_threads,
        "duckdb_memory_limit_bytes": duckdb_memory_limit_bytes,
        "small_file_threshold_bytes": small_file_threshold_bytes,
    }
