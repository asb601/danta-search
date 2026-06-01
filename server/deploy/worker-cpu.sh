#!/usr/bin/env bash
# ============================================================================
# worker-cpu.sh — CPU lane worker (prefork).
# ----------------------------------------------------------------------------
# Runs the CPU-bound ingestion stages: clean + parquet. These are RAM/core
# bound (Polars/PyArrow/DuckDB), so they want PROCESS parallelism -> -P prefork.
#
# NO -c / --concurrency flag: worker_concurrency is decided at boot by
# app/services/resource_profile.py + compute_ingestion_knobs() and surfaced via
# CELERY_WORKER_CONCURRENCY. Resize the VM (vmss.bicep sku) to scale a node;
# the worker re-tunes itself. Add VMs to scale the fleet (autoscale.sh).
#
# REQUIRES a task_routes entry in app/worker/celery_app.py (the parent will add
# it) so the CPU stages land on the "ingest_cpu" queue this worker consumes.
# Until that lands, this worker can run against the existing ingest_normal queue.
#   ASSUMED CPU-LANE ROUTING (task name -> queue):
#     gchat.ingest.clean    -> ingest_cpu
#     gchat.ingest.parquet  -> ingest_cpu
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # -> server/

CPU_QUEUES="${CPU_QUEUES:-ingest_cpu}"

exec uv run celery -A app.worker.celery_app worker \
  -P prefork \
  -Q "$CPU_QUEUES" \
  -n "gchat-cpu@%h"
