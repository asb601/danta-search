"""Phase 7 — G6 conformed-dimension resolver (the global-slicer safety gate).

A board-level categorical slicer may ONLY use a CONFORMED dimension: the SAME
business dimension present across >=2 of the board's tables with the same meaning
and member set. Filtering by a non-conformed key (e.g. "Region" = territory in one
file, postcode in another) silently filters one widget and no-ops another → an
inconsistent board. This is the dashboard-level twin of the cross-file join landmine.

Pure, fail-closed, data-driven (exact ingestion semantic_role equality + observed
member overlap — never column-name guessing):
- role_kind must be a sliceable `dimension`;
- exact `semantic_role` equality is the conformance key (two tables may name it
  `region` / `region_name` but share `custom:attribute:region`);
- the observed member set must be COMPLETE, not a truncated top-N — `top_values` is
  a sample top-N capped at ~12, so we require `len(top_values) >= cardinality` (this
  naturally limits slicers to small-domain dimensions, the right population) AND a
  readable cardinality ceiling;
- pairwise top_values Jaccard >= the policy referential floor (`min_join_overlap`),
  the same constant `join_gate.safe_join` uses — not a new dashboard magic number;
- any missing/empty/truncated/unknown signal => NOT conformed.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.services.semantic_policy import get_semantic_policy

_MAX_SLICER_CARDINALITY = 50  # a 50+-member "dimension" is not a readable slicer


@dataclass
class ConformedDimension:
    semantic_role: str            # the conformance key, e.g. custom:attribute:region
    label: str                    # display label
    column_by_table: dict         # table_name -> physical column name in that table
    tables: list                  # table_names carrying the dimension
    values: list                  # union of observed members (the slicer options)
    min_jaccard: float            # the proven minimum pairwise overlap


def _role_label(role: str) -> str:
    slug = str(role).split(":")[-1]
    return slug.replace("_", " ").strip().title() or str(role)


def _norm_values(values) -> set:
    return {str(v).strip().casefold() for v in (values or []) if str(v).strip()}


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    union = len(a | b)
    return len(a & b) / union if union else 0.0


def _is_complete_small_domain(col) -> bool:
    """top_values must BE the full member set (not a truncated sample top-N) and
    fit the readability ceiling. Fail-closed when cardinality is unknown."""
    card = getattr(col, "cardinality", None)
    top = getattr(col, "top_values", None) or []
    if card is None or not top:
        return False
    return len(top) >= card and card <= _MAX_SLICER_CARDINALITY


def resolve_conformed_dimensions(catalog) -> list:
    """Return the dimensions that are safe board-level global slicers. Pure; [] when
    the catalog is empty or nothing conforms."""
    if not catalog:
        return []
    floor = get_semantic_policy().min_join_overlap

    # Group sliceable dimension columns by EXACT semantic_role, one entry per table.
    by_role: dict = {}
    for t in catalog:
        for c in getattr(t, "columns", None) or []:
            role = getattr(c, "semantic_role", None)
            if not role or getattr(c, "role_kind", None) != "dimension":
                continue
            tname = getattr(t, "table_name", None)
            if tname is None:
                continue
            by_role.setdefault(role, {}).setdefault(tname, c)  # first column per table wins

    out: list = []
    for role, per_table in by_role.items():
        if len(per_table) < 2:                       # a global slicer needs >=2 tables
            continue
        cols = list(per_table.items())               # [(table_name, col), ...]
        if not all(_is_complete_small_domain(c) for _, c in cols):
            continue
        value_sets = [_norm_values(c.top_values) for _, c in cols]
        min_j = 1.0
        for i in range(len(value_sets)):
            for j in range(i + 1, len(value_sets)):
                min_j = min(min_j, _jaccard(value_sets[i], value_sets[j]))
        if min_j < floor:                            # fail-closed: unproven overlap
            continue

        union_vals: list = []
        seen: set = set()
        for _, c in cols:
            for v in (c.top_values or []):
                k = str(v).strip().casefold()
                if k and k not in seen:
                    seen.add(k)
                    union_vals.append(v)
        out.append(
            ConformedDimension(
                semantic_role=role,
                label=_role_label(role),
                column_by_table={tname: col.name for tname, col in cols},
                tables=[tname for tname, _ in cols],
                values=union_vals,
                min_jaccard=round(min_j, 3),
            )
        )
    return out


def resolve_widget_filters(intent, conformed, global_filters) -> tuple:
    """For ONE widget, split the active board global_filters into the ones that APPLY
    (the widget's table carries the conformed dimension — bound to its physical column)
    and the ones it is NOT AFFECTED by. Returns (applied, not_affected_labels) where
    applied = [{label, column, values}]. Honest: a widget the slicer can't touch is
    marked, never silently shown as if filtered. Pure."""
    table = ((getattr(intent, "spec", None) or {}).get("planned") or {}).get("table") \
        or (getattr(intent, "hints", None) or {}).get("table")
    by_key: dict = {}
    for d in (conformed or []):
        by_key[d.semantic_role] = d
        by_key[d.label] = d
    applied: list = []
    not_affected: list = []
    for f in (global_filters or []):
        dim = (f or {}).get("dimension")
        values = (f or {}).get("values") or []
        d = by_key.get(dim)
        if not d or not values:
            continue
        col = d.column_by_table.get(table) if table else None
        if col:
            applied.append({"label": d.label, "column": col, "values": list(values)})
        else:
            not_affected.append(d.label)
    return applied, not_affected
