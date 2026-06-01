#!/usr/bin/env bash
# ============================================================================
# worker-io.sh — IO lane worker (gevent).
# ----------------------------------------------------------------------------
# Runs the IO-bound ingestion stages: ai_description, embedding, opensearch,
# erp_classification, and the semantic enrichment/ontology stages. These are
# ~97% network wait (Azure OpenAI + OpenSearch), so they want GREENLET
# concurrency -> -P gevent, which parks thousands of coroutines on a few cores
# instead of forking a process per in-flight call.
#
# Concurrency is gevent greenlets, not processes; --concurrency here is the
# greenlet pool size and is sourced from CELERY_WORKER_CONCURRENCY / the
# resource formula's io_concurrency knob (CPU-driven fan-out, optionally clamped
# by the LLM ingest_rpm quota). We pass it via env so the box still self-sizes.
#
# REQUIRES a task_routes entry in app/worker/celery_app.py (the parent will add
# it) so these stages land on the "ingest_io" queue this worker consumes.
#   ASSUMED IO-LANE ROUTING (task name -> queue):
#     gchat.ingest.ai_description       -> ingest_io
#     gchat.ingest.embedding            -> ingest_io
#     gchat.ingest.opensearch          -> ingest_io
#     gchat.ingest.erp_classification   -> ingest_io
#     gchat.ingest.ontology             -> ingest_io
#     gchat.ingest.semantic_enrichment  -> ingest_io
# ============================================================================
set -euo pipefail

cd "$(dirname "$0")/.."   # -> server/

IO_QUEUES="${IO_QUEUES:-ingest_io}"
# gevent pool size: prefer the formula's io_concurrency; fall back to 64.
IO_CONCURRENCY="${CELERY_WORKER_CONCURRENCY:-64}"

exec uv run celery -A app.worker.celery_app worker \
  -P gevent \
  -c "$IO_CONCURRENCY" \
  -Q "$IO_QUEUES" \
  -n "gchat-io@%h"
