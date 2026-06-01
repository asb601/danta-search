#!/usr/bin/env bash
# ============================================================================
# autoscale.sh — Azure Monitor autoscale for vmss-gchat-worker, driven by the
# custom metric celery_queue_depth (published by publish_queue_depth.py).
# ----------------------------------------------------------------------------
# Design: FAST scale-OUT, SLOW scale-IN (anti-flap). The queue-depth target
# (~20 waiting tasks per instance) is the ONE number that must agree with
# ML_OPS's capacity model: target ~= 600 / per_file_seconds * per_instance_concurrency.
#
# Idempotent: re-running create on an existing profile/rule updates it.
# Prereq: vmss.bicep deployed; its managed identity has "Monitoring Metrics
# Publisher" on the VMSS; publish_queue_depth.py is emitting the metric.
# ============================================================================
set -euo pipefail

RG="${RG:-rg-danta-ingest}"
VMSS="${VMSS:-vmss-gchat-worker}"
AUTOSCALE_NAME="${AUTOSCALE_NAME:-as-gchat-worker}"
METRIC="celery_queue_depth"
NAMESPACE="danta/ingest"

# Resolve the scale-set resource id the rules attach to.
RES_ID="$(az vmss show -g "$RG" -n "$VMSS" --query id -o tsv)"

# ---- Autoscale profile: floor 1, ceiling 40 --------------------------------
# min 1 keeps the broker reachable AND keeps the queue-depth metric publisher
# (systemd timer on instance 0) alive. max 40 is the cost guardrail. Set
# min 0 only if you accept cold-start lag on the first burst.
az monitor autoscale create -g "$RG" --resource "$RES_ID" \
  --name "$AUTOSCALE_NAME" --min-count 1 --max-count 40 --count 1

# ---- SCALE OUT (aggressive) ------------------------------------------------
# >20 waiting tasks PER INSTANCE for 1m  ->  +4 instances, 2m cooldown.
# ~20/inst is roughly a 10-minute drain at the CPU-lane per-file median. Reacts
# inside one evaluation window and keeps adding +4 until per-instance backlog
# falls under 20.
az monitor autoscale rule create -g "$RG" --autoscale-name "$AUTOSCALE_NAME" \
  --condition "${METRIC} > 20 avg 1m where queue == all" \
  --scale out 4 --cooldown 2

# ---- BURST (a big dump must not climb in slow +4 steps) --------------------
# >200 waiting for 1m  ->  DOUBLE the fleet (+100%), 2m cooldown. A 1000-file
# dump doubles repeatedly (1->2->4->8...) until ML_OPS desired_replicas is met
# or the max-count=40 ceiling clamps it.
az monitor autoscale rule create -g "$RG" --autoscale-name "$AUTOSCALE_NAME" \
  --condition "${METRIC} > 200 avg 1m where queue == all" \
  --scale out 100% --cooldown 2

# ---- SCALE IN (conservative) -----------------------------------------------
# <5 waiting for a sustained 10m  ->  -1 instance, 10m cooldown. Slow single-
# step drain-down so a brief lull never tears the fleet apart. The fast-out /
# slow-in asymmetry IS the anti-flap design.
az monitor autoscale rule create -g "$RG" --autoscale-name "$AUTOSCALE_NAME" \
  --condition "${METRIC} < 5 avg 10m where queue == all" \
  --scale in 1 --cooldown 10

echo "autoscale '${AUTOSCALE_NAME}' configured on ${RES_ID} (metric ${NAMESPACE}/${METRIC})"
