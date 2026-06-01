"""Event Grid BlobCreated -> enqueue the existing ingestion pipeline.

This is the serverless IO lane front door. It does NO heavy work: it derives the
file_id from the uploaded blob's URL and hands the file to the SAME Celery task
the API already uses — ``run_ingest_pipeline`` (task name ``gchat.ingest_pipeline``,
defined in ``app/worker/ingest_tasks.py``). The Celery routing in celery_app.py
then fans the staged work across the autoscaled VMSS workers.

Keeping the function thin means scale-to-zero works cleanly: with no uploads,
no events arrive and the function app drains to 0 instances.

Required app setting: REDIS_URL (the Celery broker). Set in eventgrid_functions.sh.
"""
from __future__ import annotations

import logging
import os
from urllib.parse import unquote, urlparse

import azure.functions as func


def _derive_file_id(blob_url: str) -> str:
    """Derive the file_id from a BlobCreated event URL.

    Uploads are stored as ``.../uploads/<file_id>/<original_name>`` (or
    ``.../uploads/<file_id>.<ext>``). The first path segment after the uploads
    container is the file_id used as the primary key by the ingestion pipeline.
    Adjust the segment math here if the upload key layout changes.
    """
    path = unquote(urlparse(blob_url).path)  # /<account-container>/uploads/<file_id>/...
    parts = [p for p in path.split("/") if p]
    if "uploads" in parts:
        idx = parts.index("uploads")
        if idx + 1 < len(parts):
            candidate = parts[idx + 1]
            # Strip an extension if the file_id is the leaf (uploads/<file_id>.csv).
            return candidate.rsplit(".", 1)[0]
    # Fallback: the leaf filename without extension.
    return parts[-1].rsplit(".", 1)[0] if parts else ""


def main(event: func.EventGridEvent) -> None:
    payload = event.get_json()
    blob_url = payload.get("url", "")
    file_id = _derive_file_id(blob_url)

    if not file_id:
        logging.warning("enrich_on_blob: could not derive file_id from url=%s", blob_url)
        return

    # Build the Celery client lazily so cold imports stay cheap and the broker
    # URL is read at invocation time from the function app settings.
    from celery import Celery

    broker = os.environ["REDIS_URL"]
    client = Celery("gchat-enrich", broker=broker)

    # Enqueue by the REAL task name registered in app/worker/ingest_tasks.py.
    # send_task avoids importing the server package into the function image.
    client.send_task("gchat.ingest_pipeline", args=[file_id], queue="ingest_normal")
    logging.info("enrich_on_blob: enqueued gchat.ingest_pipeline file_id=%s", file_id)
