# danta-search — Huge-Client Ingestion Autoscale (deploy/)

Copy-pasteable infra for the burst-ingestion design: a **self-tuning VMSS worker
fleet** (CPU lane) + a **serverless Event Grid / Functions IO lane**, autoscaled
on Redis queue depth. Adapted from `huge_clients.txt` Sections 5 (IaC) and 9
(model pool / quotas).

> Nothing here edits the app. `celery_app.py` is untouched — see
> **[Required `task_routes` change](#required-task_routes-change-parent-task)**
> for the one entry the parent must add to enable the prefork/gevent split.

## Files

| File | What it is |
|---|---|
| `vmss.bicep` | Worker VM Scale Set: SystemAssigned identity, capacity floor 1, rolling upgrade, parameterized SKU, cloud-init `customData`. |
| `cloud-init.yaml` | Per-node config: installs uv, pulls the repo, drops the systemd worker unit (NO `-c` flag — self-tuned) + the queue-depth metric timer. |
| `publish_queue_depth.py` | Azure Monitor custom-metric publisher. Redis `LLEN` over `ingest_high/normal/low` -> `celery_queue_depth` via `ManagedIdentityCredential`. |
| `autoscale.sh` | `az monitor autoscale` profile + scale-out / burst / scale-in rules (min 1, max 40). |
| `eventgrid_functions.sh` | Elastic Premium plan + Function app (Python 3.12) + Event Grid system topic + BlobCreated subscription filtered to `uploads`. |
| `function_app/` | Functions skeleton: `host.json` (dynamic concurrency), `enrich_on_blob/` (Event Grid trigger) that calls `gchat.ingest_pipeline.delay()`. |
| `worker-cpu.sh` / `worker-io.sh` | The prefork/gevent lane split (see routing below). |

## Naming

`rg-danta-ingest` · `vmss-gchat-worker` · `stgchatdata` · `func-gchat-enrich`
· `plan-gchat-enrich` · `redis-gchat` (Azure Cache, db0 broker).

---

## Why ZERO per-node tuning

The worker carries no `-c` concurrency flag. On boot,
`app/services/resource_profile.py` reads cgroup/sysconf and
`compute_ingestion_knobs()` decides `worker_concurrency`, parquet/preprocess
concurrency, DuckDB threads/memory for **the box it actually landed on**. A
`D4s_v5` and a `D16s_v5` use the *same image* and both "just work". To scale a
node, change the `sku` param in `vmss.bicep`; to scale the fleet, let
`autoscale.sh` add VMs.

`task_acks_late` + `task_reject_on_worker_lost` are already set in
`celery_app.py`, so a VM killed on scale-in returns its in-flight task to Redis.

---

## Required `task_routes` change (parent task)

`worker-cpu.sh` and `worker-io.sh` split work into a **prefork CPU lane** and a
**gevent IO lane**. That requires routing the real ingestion task names (from
`app/services/ingestion_config.py :: INGEST_STAGE_SPECS`) onto two new queues.
The parent must replace the body of `celery_task_routes()` in
`app/worker/celery_app.py` (currently it routes *everything* to
`INGEST_NORMAL_QUEUE`) with this map:

```python
def celery_task_routes() -> dict[str, dict[str, str]]:
    return {
        # ---- CPU lane (prefork): RAM/core-bound, Polars/PyArrow/DuckDB ----
        "gchat.ingest.clean":               {"queue": "ingest_cpu"},
        "gchat.ingest.parquet":             {"queue": "ingest_cpu"},

        # ---- IO lane (gevent): ~97% network wait, Azure OpenAI / OpenSearch ----
        "gchat.ingest.ai_description":      {"queue": "ingest_io"},
        "gchat.ingest.embedding":           {"queue": "ingest_io"},
        "gchat.ingest.opensearch":          {"queue": "ingest_io"},
        "gchat.ingest.erp_classification":  {"queue": "ingest_io"},
        "gchat.ingest.ontology":            {"queue": "ingest_io"},
        "gchat.ingest.semantic_enrichment": {"queue": "ingest_io"},

        # ---- Local / DB-bound stages: keep on the default normal queue ----
        "gchat.ingest.metadata":            {"queue": "ingest_normal"},
        "gchat.ingest.analytics":           {"queue": "ingest_normal"},
        "gchat.ingest.relationships":       {"queue": "ingest_normal"},
        "gchat.ingest.semantic_layer":      {"queue": "ingest_normal"},
        "gchat.ingest.complete":            {"queue": "ingest_normal"},

        # ---- Orchestrators / control tasks: normal queue ----
        "gchat.ingest_pipeline":            {"queue": "ingest_normal"},
        "gchat.ingest.scoped_reprocess":    {"queue": "ingest_normal"},
        "gchat.semantic.rebuild_container": {"queue": "ingest_normal"},
    }
```

CPU stages = `clean`, `parquet` (the only members of
`_PREPROCESSING_STAGES`). IO stages = `ai_description`, `embedding`,
`opensearch`, `erp_classification`, `ontology`, `semantic_enrichment` (the
network-bound enrichment lane that races the LLM quota — Section 9). The
remaining stages are local/DB work and stay on `ingest_normal`.

> Until the parent lands this, the single cloud-init worker (consuming
> `ingest_high,ingest_normal,ingest_low`) ingests everything correctly — the
> lane split is a throughput optimization, not a correctness requirement. After
> it lands, run `worker-cpu.sh` and `worker-io.sh` as separate systemd units
> (or separate VMSS images) and have `publish_queue_depth.py` also `LLEN`
> `ingest_cpu` and `ingest_io`.

---

## Deploy order (idempotent)

```bash
# 1. Resource group
az group create -n rg-danta-ingest -l eastus

# 2. Provision Redis (Azure Cache) + storage stgchatdata if absent.
#    REDIS_URL must be the rediss://...:6380/0 broker URL.

# 3. Worker fleet (substitute your SSH key + subnet id)
az deployment group create -g rg-danta-ingest --template-file vmss.bicep \
  --parameters redisUrl="rediss://:pwd@redis-gchat:6380/0" \
               adminSshPublicKey="$(cat ~/.ssh/id_ed25519.pub)" \
               subnetId="/subscriptions/.../subnets/snet-ingest"

# 4. Grant the VMSS managed identity "Monitoring Metrics Publisher" on the VMSS
PRINCIPAL=$(az deployment group show -g rg-danta-ingest -n vmss \
  --query properties.outputs.vmssIdentityPrincipalId.value -o tsv)
RES_ID=$(az vmss show -g rg-danta-ingest -n vmss-gchat-worker --query id -o tsv)
az role assignment create --assignee "$PRINCIPAL" \
  --role "Monitoring Metrics Publisher" --scope "$RES_ID"

# 5. Autoscale rules
bash autoscale.sh

# 6. Serverless IO lane (Event Grid + Functions), then publish the code
REDIS_URL="rediss://:pwd@redis-gchat:6380/0" bash eventgrid_functions.sh
(cd function_app && func azure functionapp publish func-gchat-enrich --python)
```

---

## Verify autoscale fired

```bash
REDIS_URL="rediss://:pwd@redis-gchat:6380/0"
RES_ID=$(az vmss show -g rg-danta-ingest -n vmss-gchat-worker --query id -o tsv)

# Backlog in the broker
redis-cli -u "$REDIS_URL" llen ingest_normal

# The custom metric is flowing
az monitor metrics list --resource "$RES_ID" --metric celery_queue_depth \
  --namespace danta/ingest --interval PT1M

# Instances climbed under load
az vmss list-instances -g rg-danta-ingest -n vmss-gchat-worker -o table

# SLA check: time from first enqueue to llen==0 should be <= 10 min.
```

If the metric is empty: confirm step 4 (role grant) and that
`gchat-queue-metric.timer` is active on instance 0
(`systemctl status gchat-queue-metric.timer`).

---

## Confirm scale-to-(near-)zero

Stop uploads. After ~10 minutes of `celery_queue_depth < 5`, the scale-in rule
fires `-1` per 10m cooldown until capacity returns to the floor (1). Function
app instances drop to 0 once no `BlobCreated` events arrive (Elastic Premium
keeps `min-instances=1` warm by default — set it to 0 to allow true zero at the
cost of cold-start lag).

---

## Rollback

```bash
RES_ID=$(az vmss show -g rg-danta-ingest -n vmss-gchat-worker --query id -o tsv)

# Freeze the fleet at a fixed size (stops all scaling action)
az monitor autoscale update -g rg-danta-ingest -n as-gchat-worker \
  --min-count 2 --max-count 2

# Or disable autoscale entirely and set count by hand
az monitor autoscale update -g rg-danta-ingest -n as-gchat-worker --enabled false
az vmss scale -g rg-danta-ingest -n vmss-gchat-worker --new-capacity 1

# Bad image: rolling upgradePolicy means redeploying the prior Bicep cycles VMs
# gradually; task_acks_late protects in-flight tasks across the cycle.

# Disable the serverless IO lane (manual POST /api/ingest still works)
az eventgrid system-topic event-subscription delete -n sub-blob-enrich \
  -g rg-danta-ingest --system-topic-name st-gchat-blob
```

---

## Control knobs -> the 10-minute SLA (tune via config/CLI, never code)

| Knob | Where | Effect |
|---|---|---|
| queue-depth target (20) | `autoscale.sh` scale-out condition | lower = VMs sooner |
| burst threshold (200) | `autoscale.sh` burst rule | lower = double earlier |
| scale-out step (+4 / +100%) | `autoscale.sh` `--scale out` | bigger = faster drain |
| out cooldown (2m) | `autoscale.sh` `--cooldown` | shorter = react faster |
| in window/cooldown (10m) | `autoscale.sh` scale-in rule | longer = less thrash |
| min/max count (1 / 40) | `autoscale.sh` create | floor cost / ceiling |
| VM SKU (D4s_v5 -> D16s_v5) | `vmss.bicep` `sku` param | bigger box, auto-tuned |
| functionAppScaleLimit (200) | `eventgrid_functions.sh` | IO-lane fan-out ceiling |

The queue-depth target is the ONE autoscale number that must agree with the
ML_OPS capacity model: `target ~= 600 / per_file_seconds * per_instance_concurrency`.

> **The real ceiling is the LLM quota, not the VMs** (Section 2/9). Above the
> gpt-4o-mini TPM budget, adding workers just produces HTTP 429s. The lever
> there is buying quota / adding model-pool deployments, not more VMs.
