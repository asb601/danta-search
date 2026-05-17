"""Semantic role registry.

This file defines role behavior, not a fixed business ontology. Concrete role
names come from tenant configuration or typed dynamic roles discovered during
ingestion, such as ``custom:entity_key:claim`` or
``custom:additive_measure:premium``.
"""
from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Literal, get_args

from app.services.semantic_policy import SemanticPolicy, get_semantic_policy

RoleKind = Literal[
    "entity_key",
    "reference_key",
    "additive_measure",
    "non_additive_measure",
    "date",
    "attribute",
]
ROLE_KINDS: tuple[str, ...] = get_args(RoleKind)

_CUSTOM_ROLE_RE = re.compile(
    r"^custom:(entity_key|reference_key|additive_measure|non_additive_measure|date|attribute):[a-z][a-z0-9_]{1,63}$"
)
_SLUG_RE = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True)
class SemanticRoleSpec:
    role: str
    description: str
    kind: RoleKind
    entity_name: str | None = None
    default_aggregation: str | None = None
    risky_single_column_join: bool = False
    priority: int = 1000


_BASE_ROLE_SPECS: tuple[SemanticRoleSpec, ...] = ()

_BASE_SPECS_BY_ROLE: MappingProxyType[str, SemanticRoleSpec] = MappingProxyType(
    {spec.role: spec for spec in _BASE_ROLE_SPECS}
)

ROLE_DEFINITIONS: MappingProxyType[str, str] = MappingProxyType(
    {spec.role: spec.description for spec in _BASE_ROLE_SPECS}
)
VALID_ROLES: tuple[str, ...] = tuple(ROLE_DEFINITIONS.keys())

ENTITY_KEY_ROLES: tuple[str, ...] = tuple(spec.role for spec in _BASE_ROLE_SPECS if spec.kind == "entity_key")
REFERENCE_KEY_ROLES: tuple[str, ...] = tuple(spec.role for spec in _BASE_ROLE_SPECS if spec.kind == "reference_key")
MEASURE_ROLES: tuple[str, ...] = tuple(
    spec.role for spec in _BASE_ROLE_SPECS if spec.kind in {"additive_measure", "non_additive_measure"}
)
DATE_ROLES: tuple[str, ...] = tuple(spec.role for spec in _BASE_ROLE_SPECS if spec.kind == "date")
ATTRIBUTE_ROLES: tuple[str, ...] = tuple(spec.role for spec in _BASE_ROLE_SPECS if spec.kind == "attribute")

JOIN_KEY_ROLES: tuple[str, ...] = ENTITY_KEY_ROLES
FINGERPRINT_KEY_ROLES: tuple[str, ...] = ENTITY_KEY_ROLES + REFERENCE_KEY_ROLES
WEAK_JOIN_ROLES: tuple[str, ...] = REFERENCE_KEY_ROLES
NEVER_FINGERPRINT_JOIN_ROLES: tuple[str, ...] = MEASURE_ROLES + DATE_ROLES + ATTRIBUTE_ROLES
RISKY_SINGLE_COLUMN_JOIN_ROLES: tuple[str, ...] = tuple(
    spec.role for spec in _BASE_ROLE_SPECS if spec.risky_single_column_join
)
METRIC_ROLES: tuple[str, ...] = MEASURE_ROLES
DIMENSION_ROLES: tuple[str, ...] = ENTITY_KEY_ROLES + REFERENCE_KEY_ROLES + DATE_ROLES + ATTRIBUTE_ROLES
ROLE_PRIORITY: tuple[str, ...] = tuple(spec.role for spec in sorted(_BASE_ROLE_SPECS, key=lambda item: item.priority))


def normalize_role_slug(value: str) -> str:
    slug = _SLUG_RE.sub("_", value.strip().lower()).strip("_")
    return slug[:64] or "field"


def make_custom_role(kind: RoleKind, label: str) -> str:
    return f"custom:{kind}:{normalize_role_slug(label)}"


def is_dynamic_role(role: str | None) -> bool:
    return bool(role) and bool(_CUSTOM_ROLE_RE.match(role))


def dynamic_role_kind(role: str | None) -> RoleKind | None:
    if not role:
        return None
    match = _CUSTOM_ROLE_RE.match(role)
    return match.group(1) if match else None  # type: ignore[return-value]


def dynamic_role_label(role: str | None) -> str | None:
    if not is_dynamic_role(role):
        return None
    return role.split(":", 2)[2]


def _spec_from_dynamic_role(role: str) -> SemanticRoleSpec | None:
    kind = dynamic_role_kind(role)
    label = dynamic_role_label(role)
    if not kind or not label:
        return None
    return SemanticRoleSpec(
        role=role,
        description=f"tenant/domain role: {label.replace('_', ' ')} ({kind})",
        kind=kind,
        entity_name=label if kind == "entity_key" else None,
        default_aggregation="SUM" if kind == "additive_measure" else None,
        risky_single_column_join=kind == "reference_key",
        priority=1500,
    )


def _spec_from_config(raw: Mapping[str, Any]) -> SemanticRoleSpec | None:
    role = str(raw.get("role") or "").strip()
    kind = str(raw.get("kind") or "").strip()
    description = str(raw.get("description") or "").strip()
    if not role or kind not in ROLE_KINDS:
        return None
    if role in _BASE_SPECS_BY_ROLE:
        return None
    normalized_role = role if role.startswith("custom:") else make_custom_role(kind, role)  # type: ignore[arg-type]
    if not is_dynamic_role(normalized_role):
        return None
    default_aggregation = raw.get("default_aggregation")
    if default_aggregation is not None:
        default_aggregation = str(default_aggregation).upper()
    return SemanticRoleSpec(
        role=normalized_role,
        description=description or f"tenant/domain role: {dynamic_role_label(normalized_role)}",
        kind=kind,  # type: ignore[arg-type]
        entity_name=str(raw.get("entity_name") or dynamic_role_label(normalized_role) or "") or None,
        default_aggregation=default_aggregation if default_aggregation in {"SUM", "AVG", "MIN", "MAX", "COUNT"} else None,
        risky_single_column_join=bool(raw.get("risky_single_column_join", kind == "reference_key")),
        priority=int(raw.get("priority") or 1500),
    )


def role_catalog(custom_config: Mapping[str, Any] | None = None) -> dict[str, SemanticRoleSpec]:
    catalog = dict(_BASE_SPECS_BY_ROLE)
    for raw in (custom_config or {}).get("roles", []) or []:
        if not isinstance(raw, Mapping):
            continue
        spec = _spec_from_config(raw)
        if spec:
            catalog[spec.role] = spec
    return catalog


def role_definitions_for_prompt(custom_config: Mapping[str, Any] | None = None) -> str:
    catalog = role_catalog(custom_config)
    return "\n".join(
        f"- {role}: {spec.description} [kind={spec.kind}]"
        for role, spec in sorted(catalog.items(), key=lambda item: item[1].priority)
    )


def valid_roles(custom_config: Mapping[str, Any] | None = None) -> tuple[str, ...]:
    return tuple(role_catalog(custom_config).keys())


def get_role_spec(
    role: str | None,
    custom_config: Mapping[str, Any] | None = None,
) -> SemanticRoleSpec | None:
    if not role:
        return None
    catalog = role_catalog(custom_config)
    if role in catalog:
        return catalog[role]
    if is_dynamic_role(role):
        return _spec_from_dynamic_role(role)
    return None


def role_kind(role: str | None) -> RoleKind | None:
    spec = get_role_spec(role)
    return spec.kind if spec else None


def is_valid_role(role: str | None, custom_config: Mapping[str, Any] | None = None) -> bool:
    return get_role_spec(role, custom_config) is not None


def is_entity_key_role(role: str | None) -> bool:
    return role_kind(role) == "entity_key"


def is_reference_key_role(role: str | None) -> bool:
    return role_kind(role) == "reference_key"


def is_relationship_role(role: str | None) -> bool:
    return role_kind(role) in {"entity_key", "reference_key"}


def is_fingerprint_key_role(role: str | None) -> bool:
    return role_kind(role) in {"entity_key", "reference_key"}


def is_never_fingerprint_join_role(role: str | None) -> bool:
    return role_kind(role) in {"additive_measure", "non_additive_measure", "date", "attribute"}


def is_metric_role(role: str | None) -> bool:
    return role_kind(role) in {"additive_measure", "non_additive_measure"}


def is_additive_measure_role(role: str | None) -> bool:
    return role_kind(role) == "additive_measure"


def is_non_additive_measure_role(role: str | None) -> bool:
    return role_kind(role) == "non_additive_measure"


def is_dimension_role(role: str | None) -> bool:
    return role_kind(role) in {"entity_key", "reference_key", "date", "attribute"}


def is_date_role(role: str | None) -> bool:
    return role_kind(role) == "date"


def is_risky_single_column_join_role(role: str | None) -> bool:
    spec = get_role_spec(role)
    return bool(spec and spec.risky_single_column_join)


def relationship_confidence_for_role(
    role: str | None,
    policy: SemanticPolicy | None = None,
) -> float:
    active_policy = policy or get_semantic_policy()
    kind = role_kind(role)
    if kind == "entity_key":
        return active_policy.strong_role_confidence
    if kind == "reference_key":
        return active_policy.weak_role_confidence
    return active_policy.default_role_confidence


def entity_name_for_role(role: str | None) -> str | None:
    spec = get_role_spec(role)
    return spec.entity_name if spec else None


def role_priority(role: str | None) -> int:
    spec = get_role_spec(role)
    return spec.priority if spec else 9999


def default_aggregation_for_role(role: str | None) -> str | None:
    spec = get_role_spec(role)
    return spec.default_aggregation if spec else None
