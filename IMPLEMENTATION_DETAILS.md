# danta-search Implementation Details

This file explains the semantic workflow assembly implementation completed during the latest stabilization pass.

The work focused on one production bottleneck: the orchestration path was still behaving too lexically. Retrieval could use embeddings, but workflow assembly after retrieval still depended too heavily on token overlap, label matching, deterministic pruning, stage early exits, and hard shortlist caps.

The implementation moves danta-search toward semantic workflow continuity while keeping the existing pipeline order intact.

## Production Files Changed

| File | What Changed |
|---|---|
| `server/app/services/workflow_capability_resolver.py` | Added `coverage_state`, removed false-health behavior, and implemented bounded semantic workflow closure. |
| `server/app/services/semantic_expansion.py` | Added fair domain expansion, activation-failure recovery handling, semantic ranking boosts, and planner continuity note rendering. |
| `server/app/retrieval/semantic_recovery.py` | Replaced recovery early exits with bounded weighted aggregation. |
| `server/app/services/workflow_topology.py` | Added bridge file path and semantic role labels to topology notes. |
| `server/app/agent/graph/graph.py` | Wired retrieval evidence, approved graph edges, semantic closure seeds, expansion, topology, and prompt context into the real agent path. |

## 1. False Health Elimination

Previously, the system could return this state when workflow activation failed:

```text
workflow_completeness = 1.0
expansion_needed = false
expansion_evidence = ['no_activated_domains']
```

That was dangerous because the planner could receive incomplete context while the orchestration layer claimed the workflow was complete.

The implementation adds `coverage_state` with these allowed values:

```text
complete
partial
activation_failed
unknown
```

Activation failure now returns:

```text
coverage_state = activation_failed
workflow_completeness = 0.0
expansion_needed = true
```

Runtime proof:

```text
AFTER_FALSE_HEALTH activation_failed 0.0 True
```

## 2. Semantic Workflow Closure

The resolver no longer activates workflow domains only through query-token overlap. It now performs bounded semantic closure using existing runtime evidence.

Signals used:

```text
entity token overlap
original shortlist semantic roles
retrieval vector/opensearch channel evidence
approved graph edges adjacent to the original shortlist
role continuity through files sharing active semantic labels
```

Safety bounds:

```text
_MAX_CLOSURE_ROUNDS = 2
_MAX_CLOSURE_DOMAINS = 16
_MAX_CLOSURE_FILE_FANOUT = 40
```

Important safety behavior:

- Graph closure is anchored to the original retrieval shortlist.
- Graph-only activations cannot recursively seed role continuity.
- Expansion-added files are visible context but cannot recursively broaden workflow closure.
- Graph-neighbor activation uses primary roles from the neighbor file.
- Embedding evidence can assist semantic activation, but it does not create joins by itself.

This allows workflow continuity such as:

```text
invoice -> vendor, payment, receipt, purchase_order
delivery/shipment -> carrier, receipt
payment reconciliation -> invoice, vendor, payment, purchase_order
```

without hardcoding ERP-specific workflows.

## 3. Embedding-Assisted Assembly

Embeddings previously stopped at retrieval. The implementation now carries retrieval-channel evidence into workflow assembly.

The graph path reads retrieval telemetry and passes it into:

```text
resolve_workflow_requirements()
decide_expansion()
semantic_recovery_retrieve()
```

Embedding-backed channels such as `vector` and `opensearch` now assist:

- semantic domain activation,
- expansion candidate ranking,
- recovery aggregation scoring.

They do not override graph governance. The system still requires semantic roles, current shortlist context, or approved graph evidence to keep expansion explainable.

## 4. Recovery Aggregation

The old semantic recovery path exited as soon as one stage returned candidates. That preserved precision but destroyed workflow breadth.

The new recovery path aggregates bounded evidence from:

```text
role_cluster
graph_topology
semantic_bridge
keyword_degraded
```

Candidates are merged, deduplicated, weighted, and ranked. Keyword evidence is low-weight fallback evidence, not a blind union.

Runtime proof:

```text
AFTER_RECOVERY aggregated:graph_topology+keyword_degraded+role_cluster+semantic_bridge
```

## 5. Fair Domain Expansion

The old expansion logic could spend two slots on the first missing domain before later domains received any coverage.

The new logic is breadth-first:

```text
Round 1: best candidate per missing domain
Round 2: secondary candidates with remaining slots
```

This prevents shortlist starvation and preserves workflow breadth.

Runtime examples:

```text
invoice_matching expands vendor, payment, receipt
po_lifecycle expands vendor, payment, invoice, receipt
payment_reconciliation expands receipt, vendor, purchase_order
delivery_status expands carrier, receipt
```

The delivery workflow still remains partial after the final safety guard. That is intentional for now: the resolver avoids pulling vendor and payment context solely through broad graph drift when the original workflow evidence is delivery/shipment oriented.

## 6. Planner World-State Preservation

The topology note previously exposed missing bridge files only as short ID prefixes. That was not enough for the planner to understand what workflow context was missing.

Topology notes can now include:

```text
bridge display name
full blob path
semantic role labels
graph confidence
```

Example runtime evidence:

```text
invoice_lines [not shortlisted]
finance/invoice_lines.parquet
roles: invoice, vendor, purchase_order
```

The expansion layer also renders a compact workflow continuity note when context is still partial. It includes missing semantic domains, candidate files, confidence, and evidence.

## 7. Runtime Validation

The modified files passed Python syntax validation:

```text
python -m py_compile server/app/services/workflow_capability_resolver.py server/app/services/semantic_expansion.py server/app/retrieval/semantic_recovery.py server/app/services/workflow_topology.py server/app/agent/graph/graph.py
```

VS Code diagnostics reported no errors in the modified production files.

The runtime probes executed the real functions:

```text
resolve_workflow_requirements()
decide_expansion()
build_workflow_topology()
semantic_recovery_retrieve()
```

Final benchmark-shaped traces:

```text
invoice_matching
before partial 0.5
missing vendor, payment, receipt
after complete 1.0
```

```text
delivery_status
before partial 0.667
missing carrier, receipt
expanded carrier, receipt
after partial 0.818
missing vendor, payment
```

```text
po_lifecycle
before partial 0.333
missing vendor, payment, invoice, receipt
after complete 1.0
```

```text
payment_reconciliation
before partial 0.5
missing purchase_order, vendor, receipt
after complete 1.0
```

## 8. What This Changes In The System

The planner now receives more complete business context before it starts reasoning. Instead of seeing only the files that matched the query lexically, it can receive workflow-adjacent context derived from semantic roles, retrieval evidence, and approved graph structure.

This reduces the risk of incomplete answers for multi-file workflows such as invoice matching, PO lifecycle analysis, payment reconciliation, and delivery delay analysis.

The implementation remains bounded and explainable. It does not add a new orchestration layer, does not hardcode ERP workflows, and does not allow embedding-only joins or uncontrolled graph traversal.
