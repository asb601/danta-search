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
    approved_join_confidence: float = 0.80
    approved_join_min_overlap: float = 0.01

    planner_join_min_confidence: float = 0.65
    planner_fast_path_confidence: float = 0.75
    graph_expand_min_confidence: float = 0.85

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
