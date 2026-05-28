"""Workflow Capability Resolver — infers semantic domain requirements for a query.

PURPOSE:
  Determines which semantic capability CLUSTERS are required to answer the query,
  which are PRESENT in the current shortlist, and which are MISSING.

  This is the foundation for adaptive expansion: the system knows WHY a domain
  is needed, not just that files are missing.

KEY CONCEPT — Semantic Domain:
  A "domain" is a cluster of files that collectively serve one business capability.
  Example: "vendor" domain = files with entity_key:vendor + files with reference_key:vendor

  Domains are derived ENTIRELY from column_semantic_roles in the catalog.
  No static ontology. No hardcoded ERP/CRM/SAP knowledge.
  Domains emerge from role graph topology, not keyword lookup.

WORKFLOW COMPLETENESS SCORE:
  ratio of required_domains covered by shortlist / total required_domains.
    1.0 = all detected domains present → no expansion needed
    0.7 = some domains missing → targeted expansion
    0.0 = no domain coverage → semantic recovery

DESIGN CONSTRAINTS:
  - Zero LLM calls. Zero DB queries. Pure structural analysis.
  - No hardcoded ERP/CRM/SAP/Oracle knowledge — generalizes to any domain.
  - Domains emerge from role graph, not static ontologies.
  - All analysis operates on in-memory catalog data already fetched.
  - Operates before hydration (lean catalog fields only).
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.core.logger import pipeline_logger


# ── Role parsing ──────────────────────────────────────────────────────────────
_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")

# Role kinds that signal this file IS the authoritative master for an entity
_ENTITY_ROLE_KINDS: frozenset[str] = frozenset({"entity_key"})

# Role kinds that signal this file TRANSACTS or REFERENCES an entity
_TRANSACTION_ROLE_KINDS: frozenset[str] = frozenset({
    "reference_key", "additive_measure", "non_additive_measure"
})

# Role kinds that signal dimensional/contextual coverage
_DIMENSION_ROLE_KINDS: frozenset[str] = frozenset({"date", "attribute"})

# Token overlap fraction required to say entity_name "activates" domain_label.
# 0.40 = at least 40% of the shorter string's tokens must overlap.
_MIN_ENTITY_OVERLAP: float = 0.40

# Workflow completeness below this threshold triggers expansion.
# 0.70 = at least 70% of required domains must be covered or we expand.
_COMPLETENESS_EXPANSION_THRESHOLD: float = 0.70

# Closure bounds. These are safety caps, not relevance thresholds: they prevent a
# single broad role label from pulling the whole catalog into the workflow state.
_MAX_CLOSURE_ROUNDS: int = 2
_MAX_CLOSURE_DOMAINS: int = 16
_MAX_CLOSURE_FILE_FANOUT: int = 40

# Retrieval channels that carry embedding/semantic evidence from the retrieval
# stage into workflow assembly. The resolver never treats these as join proof;
# they only help decide which semantic domains are plausible workflow context.
_SEMANTIC_RETRIEVAL_CHANNELS: frozenset[str] = frozenset({"vector", "opensearch"})


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class SemanticDomain:
    """One business capability cluster derived from semantic roles."""
    domain_label: str            # role label, e.g. "vendor", "purchase_order"
    role_type: str               # "entity" | "transaction" | "dimension"
    all_file_ids: list[str]      # all catalog files covering this domain
    shortlist_file_ids: list[str]  # shortlisted files covering this domain
    coverage_score: float        # shortlist_coverage / all_coverage, [0.0, 1.0]
    activated_by: str            # which query entity triggered this domain
    activation_evidence: list[str] = field(default_factory=list)
    affinity_score: float = 0.0

    @property
    def is_covered(self) -> bool:
        return len(self.shortlist_file_ids) > 0

    @property
    def best_candidates(self) -> list[str]:
        """File_ids in all_file_ids that are NOT already in the shortlist."""
        short_set = set(self.shortlist_file_ids)
        return [fid for fid in self.all_file_ids if fid not in short_set]


@dataclass
class WorkflowRequirements:
    """
    Complete picture of what semantic domains are needed vs. present.

    Produced once per request after initial shortlist assembly.
    Consumed by adaptive expansion to decide whether and how to expand.
    """
    all_detected_domains: list[SemanticDomain]
    shortlist_domains: list[SemanticDomain]     # domains with ≥1 file in shortlist
    missing_domains: list[SemanticDomain]       # detected but not covered
    workflow_completeness: float                # covered / total, [0.0, 1.0]
    expansion_needed: bool
    expansion_evidence: list[str]              # human-readable trigger reasons
    coverage_state: str = "unknown"            # complete | partial | activation_failed | unknown

    def to_dict(self) -> dict:
        return {
            "total_domains": len(self.all_detected_domains),
            "covered_domains": len(self.shortlist_domains),
            "missing_domains": [
                {
                    "label": d.domain_label,
                    "type": d.role_type,
                    "activated_by": d.activated_by,
                    "candidates_available": len(d.best_candidates),
                }
                for d in self.missing_domains
            ],
            "workflow_completeness": round(self.workflow_completeness, 3),
            "expansion_needed": self.expansion_needed,
            "expansion_evidence": self.expansion_evidence,
            "coverage_state": self.coverage_state,
        }


# ── Public API ─────────────────────────────────────────────────────────────────

def resolve_workflow_requirements(
    entity_resolution: dict,           # entity_name → list[EntityCandidate]
    full_catalog: list[dict],           # lean+enriched catalog (needs column_semantic_roles)
    current_shortlist: list[dict],      # current shortlist (file_id present)
    *,
    query_text: str = "",
    retrieval_channels: dict[str, list[str]] | None = None,
    retrieval_candidate_ids: set[str] | list[str] | None = None,
    approved_edges: list | None = None,
    closure_seed_ids: set[str] | list[str] | None = None,
) -> WorkflowRequirements:
    """
    Derive workflow capability requirements from semantic roles.

    Algorithm:
      1. For each file in full_catalog, extract (domain_label, role_kind, file_id)
         triples from column_semantic_roles.
      2. For each entity in entity_resolution, find domain_labels with token overlap
         ≥ _MIN_ENTITY_OVERLAP → those domains are "activated" by the query.
      3. For each activated domain, check which files are in current_shortlist.
      4. Return WorkflowRequirements with missing domains + expansion signal.

    Returns a WorkflowRequirements with expansion_needed=False if:
      - No entities were resolved (entity_resolution empty)
      - column_semantic_roles not populated in catalog
      - All activated domains are already covered by the shortlist
    """
    # ── Build domain map from catalog ─────────────────────────────────────────
    # domain_label → {role_type → [file_id]}
    domain_map: dict[str, dict[str, list[str]]] = {}
    # file_id → [(domain_label, role_type, role_kind)]
    file_domains: dict[str, list[tuple[str, str, str]]] = {}
    file_to_blob: dict[str, str] = {}
    file_to_text: dict[str, str] = {}

    for entry in full_catalog:
        fid = entry.get("file_id")
        if not fid:
            continue
        file_to_blob[fid] = entry.get("blob_path") or fid
        file_to_text[fid] = " ".join(filter(None, [
            str(entry.get("blob_path") or ""),
            str(entry.get("ai_description") or ""),
            " ".join(str(v) for v in (entry.get("good_for") or [])),
        ])).lower()
        roles: dict = entry.get("column_semantic_roles") or {}
        for _, role_str in roles.items():
            parsed = _parse_role(str(role_str) if role_str else None)
            if not parsed:
                continue
            kind, label = parsed

            rtype = _role_type_for_kind(kind)
            if not rtype:
                continue

            bucket = domain_map.setdefault(label, {}).setdefault(rtype, [])
            if fid not in bucket:
                bucket.append(fid)
            triple = (label, rtype, kind)
            if triple not in file_domains.setdefault(fid, []):
                file_domains[fid].append(triple)

    if not domain_map:
        # column_semantic_roles not populated — no domain analysis possible
        return WorkflowRequirements(
            all_detected_domains=[],
            shortlist_domains=[],
            missing_domains=[],
            workflow_completeness=0.0,
            expansion_needed=False,
            expansion_evidence=["no_semantic_roles_in_catalog"],
            coverage_state="unknown",
        )

    shortlist_ids: set[str] = {e.get("file_id") for e in current_shortlist if e.get("file_id")}
    retrieval_channels = retrieval_channels or {}
    retrieval_candidate_id_set: set[str] = set(retrieval_candidate_ids or [])
    edge_pairs = _normalise_approved_edges(approved_edges or [])
    graph_seed_ids: set[str] = set(closure_seed_ids or shortlist_ids)

    activated: dict[tuple[str, str], dict] = {}

    def _activate(
        domain_label: str,
        role_type: str,
        activated_by: str,
        evidence: str,
        affinity: float,
    ) -> bool:
        """Activate one domain if it exists in the catalog. Returns True if new."""
        file_ids = domain_map.get(domain_label, {}).get(role_type, [])
        if not file_ids:
            return False
        key = (domain_label, role_type)
        is_new = key not in activated
        state = activated.setdefault(key, {
            "activated_by": activated_by,
            "evidence": [],
            "affinity": 0.0,
        })
        if evidence not in state["evidence"]:
            state["evidence"].append(evidence)
        state["affinity"] = max(float(state.get("affinity") or 0.0), affinity)
        if is_new and len(activated) > _MAX_CLOSURE_DOMAINS:
            activated.pop(key, None)
            return False
        return is_new

    def _activate_file_domains(
        fid: str,
        activated_by: str,
        evidence_prefix: str,
        affinity: float,
        *,
        include_dimensions: bool,
        primary_only: bool = False,
    ) -> set[tuple[str, str]]:
        new_keys: set[tuple[str, str]] = set()
        for label, rtype, kind in file_domains.get(fid, []):
            if rtype == "dimension" and not include_dimensions:
                continue
            if primary_only and not _label_matches_file(label, file_to_text.get(fid, "")):
                continue
            evidence = f"{evidence_prefix}:{file_to_blob.get(fid, fid)}:{kind}:{label}"
            if _activate(label, rtype, activated_by, evidence, affinity):
                new_keys.add((label, rtype))
            if len(activated) >= _MAX_CLOSURE_DOMAINS:
                break
        return new_keys

    # ── Activate domains via entity_resolution ─────────────────────────────────
    for entity_name, candidates in entity_resolution.items():
        if not candidates:
            continue
        for domain_label, role_files in domain_map.items():
            overlap = _token_overlap(entity_name, domain_label)
            if overlap < _MIN_ENTITY_OVERLAP:
                continue
            for role_type, file_ids in role_files.items():
                if not file_ids:
                    continue
                _activate(
                    domain_label,
                    role_type,
                    activated_by=entity_name,
                    evidence=f"entity_token_overlap:{entity_name}->{domain_label}",
                    affinity=overlap,
                )

    # ── Carry retrieval semantic evidence into workflow assembly ─────────────
    # A file already in the shortlist is planner-visible context. Its semantic
    # roles can activate adjacent domains; vector/opensearch channels mark that
    # this context was surfaced by embedding retrieval rather than lexical match.
    for fid in list(shortlist_ids)[:_MAX_CLOSURE_FILE_FANOUT]:
        is_original_seed = fid in graph_seed_ids
        channels = set(retrieval_channels.get(fid, []) or [])
        semantic_channels = sorted(channels & _SEMANTIC_RETRIEVAL_CHANNELS)
        prefix = "embedding_retrieval" if semantic_channels else "shortlist_context"
        affinity = 0.85 if semantic_channels else 0.65
        _activate_file_domains(
            fid,
            activated_by=(
                "+".join(semantic_channels) if is_original_seed and semantic_channels
                else "shortlist" if is_original_seed
                else "expanded_context"
            ),
            evidence_prefix=prefix,
            affinity=affinity,
            include_dimensions=True,
            primary_only=not is_original_seed,
        )

    # ── Semantic workflow closure ────────────────────────────────────────────
    # Closure is structural and bounded. It expands from activated domains to
    # adjacent domains through role co-occurrence, retrieval semantic proximity,
    # and approved graph neighbors. It does not invent joins and does not walk
    # recursively through the entire graph.
    frontier: set[tuple[str, str]] = set(activated)
    for _ in range(_MAX_CLOSURE_ROUNDS):
        if not frontier or len(activated) >= _MAX_CLOSURE_DOMAINS:
            break

        continuity_frontier = {
            key for key in frontier
            if str(activated.get(key, {}).get("activated_by") or "")
            not in {"approved_graph", "expanded_context"}
        }
        active_labels = {label for label, _ in continuity_frontier}
        active_files: set[str] = set(shortlist_ids)
        for label, rtype in continuity_frontier:
            active_files.update(domain_map.get(label, {}).get(rtype, [])[:_MAX_CLOSURE_FILE_FANOUT])

        # Graph continuity is anchored to the current shortlist only. This keeps
        # approved topology useful without letting newly inferred domains chain
        # into unrelated graph neighborhoods before expansion has been bounded.
        graph_neighbor_ids = _graph_neighbors(graph_seed_ids, edge_pairs)
        candidate_pool = set(shortlist_ids) | retrieval_candidate_id_set | graph_neighbor_ids

        next_frontier: set[tuple[str, str]] = set()

        # Role continuity: files that contain an already-active label can reveal
        # the next workflow domain in the same business process.
        continuity_files: set[str] = set()
        for label in active_labels:
            for files in domain_map.get(label, {}).values():
                continuity_files.update(files)
        bounded_continuity = (continuity_files & candidate_pool) or continuity_files
        for fid in list(bounded_continuity)[:_MAX_CLOSURE_FILE_FANOUT]:
            next_frontier.update(_activate_file_domains(
                fid,
                activated_by="role_continuity",
                evidence_prefix="role_continuity",
                affinity=0.72,
                include_dimensions=fid in shortlist_ids,
            ))
            if len(activated) >= _MAX_CLOSURE_DOMAINS:
                break

        # Approved graph continuity: relationship edges can surface neighbor
        # files as workflow context, but still only as bounded activation input.
        for fid in list(graph_neighbor_ids)[:_MAX_CLOSURE_FILE_FANOUT]:
            next_frontier.update(_activate_file_domains(
                fid,
                activated_by="approved_graph",
                evidence_prefix="approved_graph",
                affinity=0.78,
                include_dimensions=False,
                primary_only=True,
            ))
            if len(activated) >= _MAX_CLOSURE_DOMAINS:
                break

        # Embedding-assisted continuity: vector/opensearch candidates may reveal
        # synonyms or adjacent domains (e.g. delivery -> shipment). They only add
        # domains when the candidate also shares an active semantic role label,
        # preventing embedding-only workflow drift.
        semantic_candidate_ids = [
            fid for fid in retrieval_candidate_id_set
            if set(retrieval_channels.get(fid, []) or []) & _SEMANTIC_RETRIEVAL_CHANNELS
        ]
        for fid in semantic_candidate_ids[:_MAX_CLOSURE_FILE_FANOUT]:
            labels = {label for label, _, _ in file_domains.get(fid, [])}
            if not labels & active_labels:
                continue
            next_frontier.update(_activate_file_domains(
                fid,
                activated_by="embedding_role_affinity",
                evidence_prefix="embedding_role_affinity",
                affinity=0.82,
                include_dimensions=False,
            ))
            if len(activated) >= _MAX_CLOSURE_DOMAINS:
                break

        frontier = next_frontier - frontier

    activated_domains = _build_semantic_domains(activated, domain_map, shortlist_ids)

    if not activated_domains:
        coverage_state = "activation_failed" if (entity_resolution or query_text.strip()) else "unknown"
        return WorkflowRequirements(
            all_detected_domains=[],
            shortlist_domains=[],
            missing_domains=[],
            workflow_completeness=0.0,
            expansion_needed=(coverage_state == "activation_failed"),
            expansion_evidence=["no_activated_domains"],
            coverage_state=coverage_state,
        )

    covered = [d for d in activated_domains if d.is_covered]
    missing = [d for d in activated_domains if not d.is_covered]
    completeness = len(covered) / max(1, len(activated_domains))

    evidence: list[str] = []
    for d in missing:
        evidence.append(
            f"missing_{d.role_type}_domain:{d.domain_label}"
            f"(activated_by:{d.activated_by},candidates:{len(d.all_file_ids)})"
        )

    if missing:
        coverage_state = "partial"
    else:
        coverage_state = "complete"

    expansion_needed = (
        coverage_state == "partial"
        and completeness < _COMPLETENESS_EXPANSION_THRESHOLD
        and any(len(d.best_candidates) > 0 for d in missing)
    )

    pipeline_logger.info(
        "workflow_requirements_resolved",
        total_domains=len(activated_domains),
        covered_domains=len(covered),
        missing_domains=len(missing),
        completeness=round(completeness, 3),
        coverage_state=coverage_state,
        expansion_needed=expansion_needed,
        evidence=evidence[:5],
    )

    return WorkflowRequirements(
        all_detected_domains=activated_domains,
        shortlist_domains=covered,
        missing_domains=missing,
        workflow_completeness=completeness,
        expansion_needed=expansion_needed,
        expansion_evidence=evidence,
        coverage_state=coverage_state,
    )


# ── Helpers ────────────────────────────────────────────────────────────────────

def _parse_role(role_str: str | None) -> tuple[str, str] | None:
    if not role_str:
        return None
    m = _ROLE_RE.match(str(role_str))
    return (m.group(1), m.group(2)) if m else None


def _role_type_for_kind(kind: str) -> str | None:
    if kind in _ENTITY_ROLE_KINDS:
        return "entity"
    if kind in _TRANSACTION_ROLE_KINDS:
        return "transaction"
    if kind in _DIMENSION_ROLE_KINDS:
        return "dimension"
    return None


def _normalise_approved_edges(rows: list) -> list[tuple[str, str, float]]:
    edges: list[tuple[str, str, float]] = []
    for row in rows:
        a = getattr(row, "file_a_id", None) or (row[0] if isinstance(row, tuple) and len(row) > 0 else None)
        b = getattr(row, "file_b_id", None) or (row[1] if isinstance(row, tuple) and len(row) > 1 else None)
        if not a or not b:
            continue
        conf = getattr(row, "confidence_score", None)
        if conf is None and isinstance(row, tuple) and len(row) > 2:
            conf = row[2]
        edges.append((str(a), str(b), float(conf if conf is not None else 0.0)))
    return edges


def _graph_neighbors(seed_ids: set[str], edges: list[tuple[str, str, float]]) -> set[str]:
    if not seed_ids:
        return set()
    neighbors: set[str] = set()
    for a, b, _ in edges:
        if a in seed_ids and b not in seed_ids:
            neighbors.add(b)
        elif b in seed_ids and a not in seed_ids:
            neighbors.add(a)
    return neighbors


def _build_semantic_domains(
    activated: dict[tuple[str, str], dict],
    domain_map: dict[str, dict[str, list[str]]],
    shortlist_ids: set[str],
) -> list[SemanticDomain]:
    domains: list[SemanticDomain] = []
    for (domain_label, role_type), state in activated.items():
        file_ids = list(domain_map.get(domain_label, {}).get(role_type, []))
        if not file_ids:
            continue
        shortlist_coverage = [fid for fid in file_ids if fid in shortlist_ids]
        coverage_score = len(shortlist_coverage) / max(1, len(file_ids))
        domains.append(SemanticDomain(
            domain_label=domain_label,
            role_type=role_type,
            all_file_ids=file_ids,
            shortlist_file_ids=shortlist_coverage,
            coverage_score=coverage_score,
            activated_by=str(state.get("activated_by") or "semantic_closure"),
            activation_evidence=list(state.get("evidence") or []),
            affinity_score=float(state.get("affinity") or 0.0),
        ))
    return domains


def _token_overlap(a: str, b: str) -> float:
    """Token overlap fraction between two label strings. Case-insensitive."""
    ta = tuple(dict.fromkeys(t for t in re.split(r"[^a-z0-9]+", a.lower()) if t))
    tb = tuple(dict.fromkeys(t for t in re.split(r"[^a-z0-9]+", b.lower()) if t))
    if not ta or not tb:
        return 0.0
    set_a = set(ta)
    set_b = set(tb)
    acronym_a = "".join(t[0] for t in ta) if len(ta) > 1 else ""
    acronym_b = "".join(t[0] for t in tb) if len(tb) > 1 else ""
    if acronym_a and acronym_a in set_b:
        return 1.0
    if acronym_b and acronym_b in set_a:
        return 1.0
    return len(set_a & set_b) / min(len(set_a), len(set_b))


def _label_matches_file(label: str, file_text: str) -> bool:
    label_tokens = [t for t in re.split(r"[^a-z0-9]+", label.lower()) if len(t) >= 3]
    text_tokens = set(t for t in re.split(r"[^a-z0-9]+", file_text.lower()) if len(t) >= 3)
    return any(t in text_tokens for t in label_tokens)
