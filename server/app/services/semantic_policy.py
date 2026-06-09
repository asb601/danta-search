"""Runtime semantic policy configuration.

Semantic roles describe meaning; semantic policy describes operational thresholds.
Keeping these values here prevents scattered magic numbers in relationship
inference, semantic approval, graph expansion, and planning.

Every value can be overridden with an environment variable prefixed by
GCHAT_SEMANTIC_, for example:
    GCHAT_SEMANTIC_PLANNER_FAST_PATH_CONFIDENCE=0.82
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from functools import lru_cache


@dataclass(frozen=True)
class SemanticPolicy:
    min_relationship_confidence: float = 0.50
    strong_role_confidence: float = 0.85
    weak_role_confidence: float = 0.55
    default_role_confidence: float = 0.70

    max_join_null_rate: float = 0.40
    min_value_overlap: float = 0.01
    inner_join_overlap: float = 0.05
    fingerprint_min_confidence: float = 0.60
    fingerprint_max_confidence: float = 0.98

    ontology_key_min_unique_rate: float = 0.01
    pk_unique_rate: float = 0.95
    pk_null_rate: float = 0.01
    generic_pk_unique_rate: float = 0.98
    generic_fk_max_null_rate: float = 0.20
    generic_fk_min_unique_rate: float = 0.02
    generic_fk_max_unique_rate: float = 0.80
    max_fingerprints_per_column: int = 1000
    min_distinct_key_values: int = 2

    entity_with_pk_confidence: float = 0.80
    entity_unknown_grain_confidence: float = 0.45
    # Lowered from 0.80: a genuine high-cardinality join key (e.g. Vendor_ID at
    # ~79-83% value overlap) must be approvable. The cardinality guard + overlap
    # floor below are the real false-positive defense, not the confidence floor.
    approved_join_confidence: float = 0.72
    approved_join_min_overlap: float = 0.01

    # Edge-creation gates (relationship_detector): a real join must clear BOTH a
    # value-overlap floor and a minimum cardinality. This rejects non-referential
    # document keys (PO_Number/Material_No ~0% overlap) and coincidental overlap
    # on tiny enums — data-driven, never by column name.
    min_join_overlap: float = 0.50
    min_join_cardinality: int = 8
    # Ubiquity ceiling for master-key promotion: a join column present in MORE
    # than this fraction of a container's files is an audit/system column (e.g.
    # created_by, last_updated_by appear in ~all tables) and is NEVER promoted to
    # an approved join — even when its cardinality matches a real master. Ubiquity
    # is the decisive separator between a business key and an equally-cardinal
    # audit column. Data-derived (computed per container), never a name list.
    ubiquity_ceiling: float = 0.60
    # Clone-overlap floor for the templated/copied-column guard. A copied column
    # (NOT a referential FK) has the verified signature value_overlap >= this AND
    # cardinality_left == cardinality_right (the two sides hold the IDENTICAL
    # generated value set). Such an edge is a template-clone document key joining
    # unrelated tables (e.g. AP_BATCHES_ALL ⋈ AP_BANK_ACCOUNTS_ALL on PO_HEADER_ID)
    # and is NEVER promoted, even on a same-name match. Real masters differ in
    # cardinality between their two sides, so they clear this guard. Distributional,
    # never a name list.
    clone_overlap_floor: float = 0.999
    # Confidence = weighted blend of value overlap and (log-scaled) cardinality,
    # so a high-overlap high-cardinality key scores above the approval floor while
    # a high-overlap tiny-domain coincidence does not.
    confidence_overlap_weight: float = 0.75
    confidence_cardinality_weight: float = 0.25
    confidence_cardinality_reference: float = 1000.0

    planner_join_min_confidence: float = 0.65
    planner_fast_path_confidence: float = 0.75
    # Aligned with approved_join_confidence (0.72): if expansion required MORE
    # confidence than approval, approved joins would never reach the retrieval
    # shortlist via one-hop expansion (silent narrowing).
    graph_expand_min_confidence: float = 0.72

    planner_metric_bonus: float = 0.30
    planner_raw_intent_bonus: float = 0.20
    planner_explicit_aggregation_bonus: float = 0.20
    planner_default_aggregation_bonus: float = 0.10
    planner_dimension_bonus: float = 0.15
    planner_time_filter_bonus: float = 0.10
    planner_missing_time_filter_penalty: float = 0.10
    planner_join_bonus: float = 0.10
    planner_missing_join_penalty: float = 0.15
    planner_vague_single_file_penalty: float = 0.10

    min_overlap_fingerprint_count: int = 2
    # Max fan-out a verified 1:N join may exhibit. fanout_estimate is 1 /
    # unique_rate[pk_side]; a true PK is ~1.0 (one row per key), so a small margin
    # above 1.0 still admits real keys with minor sample duplication while
    # rejecting many-to-many noise. Value-derived per pair, never a name list.
    # Env-overridable as GCHAT_SEMANTIC_MAX_JOIN_FANOUT.
    max_join_fanout: float = 1.5
    # RESOLVE-contract scoring knobs (v2 query lane). The caller injects these
    # into the Contract / needs_fallback so no tuned constant lives in scoring
    # code. unverified_twin_penalty: confidence multiplier when a twin cluster
    # has no verified master. resolve_contract_tau: confidence cutoff below which
    # the runtime drops to the agent fallback. Value-/policy-driven, not name
    # lists. Env-overridable as GCHAT_SEMANTIC_UNVERIFIED_TWIN_PENALTY /
    # GCHAT_SEMANTIC_RESOLVE_CONTRACT_TAU.
    unverified_twin_penalty: float = 0.5
    resolve_contract_tau: float = 0.75
    relation_direct_limit: int = 20
    relation_max_hops: int = 4
    relation_max_paths: int = 8
    relation_expand_edge_limit: int = 400


def _env_name(field_name: str) -> str:
    return f"GCHAT_SEMANTIC_{field_name.upper()}"


def _coerce_env_value(raw: str, current_value: object) -> object:
    if isinstance(current_value, int) and not isinstance(current_value, bool):
        return int(raw)
    if isinstance(current_value, float):
        return float(raw)
    return raw


@lru_cache(maxsize=1)
def get_semantic_policy() -> SemanticPolicy:
    defaults = SemanticPolicy()
    overrides: dict[str, object] = {}
    for field in fields(defaults):
        current_value = getattr(defaults, field.name)
        raw = os.getenv(_env_name(field.name))
        if raw is None or raw == "":
            continue
        overrides[field.name] = _coerce_env_value(raw, current_value)
    return SemanticPolicy(**overrides)


def reset_semantic_policy_cache() -> None:
    get_semantic_policy.cache_clear()
