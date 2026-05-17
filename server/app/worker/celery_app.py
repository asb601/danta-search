"""
Celery application — broker and worker configuration.

Queue layout (3 queues, priority order):
  ingest_high   — reserved for future lightweight status-critical tasks
  ingest_normal — heavy staged ingestion work
  ingest_low    — reserved for future deferred refinements/backfills

Worker start commands (run from server/):
  # Normal worker — heavy stage tasks. Keep concurrency conservative:
  # each slot may run preprocessing or Parquet conversion for a large file.
  uv run celery -A app.worker.celery_app worker -Q ingest_normal -c 2 -n ingest_normal@%h

  # Optional high-priority worker — reserved for future split tasks only.
  uv run celery -A app.worker.celery_app worker -Q ingest_high -c 2 -n ingest_high@%h

  # Low worker — 2 concurrent slots (analytics are cheap, don't over-allocate RAM)
  uv run celery -A app.worker.celery_app worker -Q ingest_low -c 2 -n ingest_low@%h

Environment variables required in .env:
  REDIS_URL         = redis://localhost:6379/0       # broker
  REDIS_URL_RESULTS = redis://localhost:6379/1       # result backend
  (Azure Cache for Redis: rediss://:PASSWORD@HOST:6380/0)
"""
from __future__ import annotations

from celery import Celery

from app.core.config import get_settings


def _make_celery() -> Celery:
    settings = get_settings()

    app = Celery("gchat")

    app.conf.update(
        # ── Broker + backend ──────────────────────────────────────────────────
        broker_url=settings.REDIS_URL,
        result_backend=settings.REDIS_URL_RESULTS,

        # ── Serialization ─────────────────────────────────────────────────────
        task_serializer="json",
        result_serializer="json",
        accept_content=["json"],

        # ── Reliability ───────────────────────────────────────────────────────
        # acks_late: task is NOT acknowledged until it completes successfully.
        # If the worker process is killed mid-task (OOM, VM restart), the
        # message goes back to the queue and another worker picks it up.
        # Without this, a crash permanently loses the task.
        task_acks_late=True,
        task_reject_on_worker_lost=True,

        # If an Azure/HTTP client hangs forever, the worker slot must be freed.
        # The task itself marks the File row failed when final retry is exhausted.
        task_soft_time_limit=60 * 45,  # 45 minutes: graceful timeout
        task_time_limit=60 * 50,       # 50 minutes: hard kill safety net

        # ── Prefetch ──────────────────────────────────────────────────────────
        # prefetch_multiplier=1: each worker slot fetches exactly one task at a
        # time. This is critical for ingest tasks (each can take minutes and use
        # hundreds of MB RAM). Without this, a worker with 4 slots would prefetch
        # 16 tasks, blocking other workers from seeing them.
        worker_prefetch_multiplier=1,

        # ── Result TTL ────────────────────────────────────────────────────────
        result_expires=86400,  # 24h — results are only used for status checks

        # Emit task lifecycle events so Flower / Celery inspect / monitoring can
        # detect stuck workers and queue buildup without extra app code.
        worker_send_task_events=True,
        task_send_sent_event=True,

        # ── Timezone ─────────────────────────────────────────────────────────
        timezone="UTC",
        enable_utc=True,

        # ── Task routing ─────────────────────────────────────────────────────
        task_routes={
          "gchat.ingest_pipeline": {"queue": "ingest_normal"},
          "gchat.ingest.clean": {"queue": "ingest_normal"},
          "gchat.ingest.parquet": {"queue": "ingest_normal"},
          "gchat.ingest.metadata": {"queue": "ingest_normal"},
          "gchat.ingest.ai_description": {"queue": "ingest_normal"},
          "gchat.ingest.ontology": {"queue": "ingest_normal"},
          "gchat.ingest.embedding": {"queue": "ingest_normal"},
          "gchat.ingest.opensearch": {"queue": "ingest_normal"},
          "gchat.ingest.analytics": {"queue": "ingest_normal"},
          "gchat.ingest.relationships": {"queue": "ingest_normal"},
          "gchat.ingest.semantic_layer": {"queue": "ingest_normal"},
          "gchat.ingest.complete": {"queue": "ingest_normal"},
        },

        # ── Default queue ────────────────────────────────────────────────────
        task_default_queue="ingest_normal",
    )

    return app


celery_app = _make_celery()
