#!/usr/bin/env python3
"""Azure Monitor custom-metric publisher for ingestion queue depth.

Reads ``LLEN`` over the real Celery ingest queues (ingest_high / ingest_normal /
ingest_low) from Redis and POSTs the total to Azure Monitor as the custom metric
``celery_queue_depth`` (namespace ``danta/ingest``). The autoscale rules in
``autoscale.sh`` scale ``vmss-gchat-worker`` on this number — Azure Monitor can
read CPU/RAM natively but NOT "files waiting in Redis", so we publish it.

Auth is the VMSS SystemAssigned identity via ``ManagedIdentityCredential`` — no
secret on disk. Grant it "Monitoring Metrics Publisher" on the scale set.

Run modes:
    python deploy/publish_queue_depth.py --once     # one POST, exits (systemd timer)
    python deploy/publish_queue_depth.py            # loop, POST every --interval s

Environment:
    REDIS_URL            broker URL, e.g. rediss://:pwd@redis-gchat:6380/0
    AZURE_REGION         region short name, e.g. eastus
    VMSS_RESOURCE_ID     /subscriptions/.../virtualMachineScaleSets/vmss-gchat-worker
    INGEST_HIGH_QUEUE    optional override (default "ingest_high")
    INGEST_NORMAL_QUEUE  optional override (default "ingest_normal")
    INGEST_LOW_QUEUE     optional override (default "ingest_low")
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time

import redis
import requests
from azure.identity import ManagedIdentityCredential

# Real queue names — must match settings.INGEST_*_QUEUE / celery_app.py routing.
QUEUES = (
    os.environ.get("INGEST_HIGH_QUEUE", "ingest_high"),
    os.environ.get("INGEST_NORMAL_QUEUE", "ingest_normal"),
    os.environ.get("INGEST_LOW_QUEUE", "ingest_low"),
)

METRIC_NAME = "celery_queue_depth"
METRIC_NAMESPACE = "danta/ingest"
_SCOPE = "https://monitoring.azure.com/.default"


def _require_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        sys.exit(f"publish_queue_depth: missing required env var {name}")
    return value


def queue_depth(client: "redis.Redis") -> int:
    """Total pending tasks across the ingest queues. A bad queue counts as 0."""
    total = 0
    for name in QUEUES:
        try:
            total += int(client.llen(name))
        except Exception:  # noqa: BLE001 - one bad queue must not drop the metric
            continue
    return max(0, total)


def publish(cred: ManagedIdentityCredential, region: str, res_id: str, depth: int) -> None:
    """POST one data point of celery_queue_depth to Azure Monitor."""
    token = cred.get_token(_SCOPE).token
    body = {
        "time": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "data": {
            "baseData": {
                "metric": METRIC_NAME,
                "namespace": METRIC_NAMESPACE,
                "dimNames": ["queue"],
                "series": [
                    {
                        "dimValues": ["all"],
                        "min": depth,
                        "max": depth,
                        "sum": depth,
                        "count": 1,
                    }
                ],
            }
        },
    }
    url = f"https://{region}.monitoring.azure.com{res_id}/metrics"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        data=json.dumps(body),
        timeout=10,
    )
    resp.raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="publish once and exit")
    parser.add_argument("--interval", type=int, default=60, help="loop interval seconds")
    args = parser.parse_args()

    redis_url = _require_env("REDIS_URL")
    region = _require_env("AZURE_REGION")
    res_id = _require_env("VMSS_RESOURCE_ID")

    cred = ManagedIdentityCredential()
    client = redis.from_url(redis_url)

    if args.once:
        publish(cred, region, res_id, queue_depth(client))
        return 0

    while True:
        try:
            publish(cred, region, res_id, queue_depth(client))
        except Exception as exc:  # noqa: BLE001 - never let a transient error kill the loop
            print(f"publish_queue_depth: transient error: {exc}", file=sys.stderr)
        time.sleep(max(5, args.interval))


if __name__ == "__main__":
    raise SystemExit(main())
