#!/usr/bin/env bash
# ============================================================================
# eventgrid_functions.sh — serverless per-file IO lane (scale-to-zero).
# ----------------------------------------------------------------------------
# Flow:  blob uploaded -> BlobCreated event -> Event Grid system topic ->
#        event subscription (filtered to the uploads container) -> Function
#        (function_app/enrich_on_blob) -> run_ingest_pipeline.delay(file_id).
#
# Elastic Premium plan = pre-warmed instances (no cold start) + large fan-out,
# which is what the IO/enrichment lane needs to race the LLM quota. The function
# itself does NO heavy work: it derives file_id and enqueues the existing Celery
# task, so the VMSS CPU lane + the model-pool IO lane do the real ingestion.
#
# Idempotent-ish: create commands are safe to re-run; publish overwrites code.
# ============================================================================
set -euo pipefail

RG="${RG:-rg-danta-ingest}"
LOC="${LOC:-eastus}"
STG="${STG:-stgchatdata}"            # storage account that holds the uploads container
FUNC="${FUNC:-func-gchat-enrich}"
PLAN="${PLAN:-plan-gchat-enrich}"
REDIS_URL="${REDIS_URL:?set REDIS_URL, e.g. rediss://:pwd@redis-gchat:6380/0}"
UPLOADS_CONTAINER="${UPLOADS_CONTAINER:-uploads}"

# ---- Elastic Premium plan: warm instances, big burst -----------------------
# EP1 base; min 1 warm instance kills cold start; max-burst 200 is the warm-
# instance ceiling for a large dump.
az functionapp plan create -g "$RG" -n "$PLAN" --location "$LOC" \
  --sku EP1 --min-instances 1 --max-burst 200 --is-linux

# ---- Function app (Python 3.12, Functions v4) ------------------------------
az functionapp create -g "$RG" -n "$FUNC" --plan "$PLAN" \
  --storage-account "$STG" --runtime python --runtime-version 3.12 \
  --functions-version 4 --os-type Linux

# functionAppScaleLimit caps total concurrent instances (IO-lane fan-out
# ceiling). REDIS_URL lets the function enqueue onto the same broker the workers
# drain. Match functionAppScaleLimit to the plan max-burst.
az functionapp config appsettings set -g "$RG" -n "$FUNC" --settings \
  functionAppScaleLimit=200 \
  REDIS_URL="$REDIS_URL"

# ---- Deploy the function code (function_app/) ------------------------------
# From server/deploy/function_app:  func azure functionapp publish "$FUNC"
echo "Publish the function code with:"
echo "  (cd \"$(dirname "$0")/function_app\" && func azure functionapp publish $FUNC --python)"

# ---- Event Grid system topic over the storage account ----------------------
STG_ID="$(az storage account show -g "$RG" -n "$STG" --query id -o tsv)"
az eventgrid system-topic create -g "$RG" -n st-gchat-blob --location "$LOC" \
  --topic-type Microsoft.Storage.StorageAccounts \
  --source "$STG_ID"

# ---- BlobCreated subscription, filtered to the uploads container -----------
# subject-begins-with restricts events to /blobServices/default/containers/<uploads>/
# so we ingest user uploads only, not Parquet outputs or other blobs.
FUNC_ID="$(az functionapp function show -g "$RG" -n "$FUNC" \
  --function-name enrich_on_blob --query id -o tsv)"
az eventgrid system-topic event-subscription create -n sub-blob-enrich -g "$RG" \
  --system-topic-name st-gchat-blob --endpoint-type azurefunction \
  --endpoint "$FUNC_ID" --included-event-types Microsoft.Storage.BlobCreated \
  --subject-begins-with "/blobServices/default/containers/${UPLOADS_CONTAINER}/"

echo "Event Grid -> ${FUNC}/enrich_on_blob wired for container '${UPLOADS_CONTAINER}'."
