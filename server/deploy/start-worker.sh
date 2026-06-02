#!/usr/bin/env bash
# ============================================================================
# start-worker.sh — SELF-TUNING ingestion worker (single-box, no lane split).
# ----------------------------------------------------------------------------
# The slow part of ingestion is the LLM enrichment (description / roles / ERP /
# embedding) — it is ~95% network wait, so it wants GREENLET concurrency, not
# processes. This launcher reads the resource formula's `io_concurrency` knob
# (= min(io_cap, io_fanout × cpu_cores), optionally clamped by the LLM
# ingest_rpm quota) and starts a gevent worker at exactly that size.
#
# DYNAMIC BY DESIGN: nothing is hardcoded. On a 2-core client VM it self-selects
# ~16; on a 10-core box ~64. Resize the VM → the worker re-tunes on next start.
# This is the "resize the box, never touch the code" rule applied to the worker.
#
# It consumes the EXISTING queues (high,normal,low) so it drains whatever is
# already enqueued — no lane-split routing required.
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # -> server/

# Pull io_concurrency straight from the formula (cgroup/spec-aware).
IO_C="$(
  uv run python -c 'from app.services.resource_profile import get_resource_profile, compute_ingestion_knobs; print(compute_ingestion_knobs(get_resource_profile())["io_concurrency"])'
)"

echo "[start-worker] self-tuned io_concurrency=${IO_C} (gevent greenlet pool) for this VM"

exec uv run celery -A app.worker.celery_app worker \
  -P gevent \
  -c "${IO_C}" \
  -Q high,normal,low \
  -n "gchat-ingest@%h" \
  --loglevel=info
