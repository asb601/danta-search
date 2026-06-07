"""Phase 2 — JOIN gate: deterministic, data-driven safety for cross-file widgets.

The dashboard layer writes zero SQL and cannot force the agent's joins, so this
module provides pure predicates that (1) decide which joins are ADVERTISED to the
planner/agent (grounding) and (3) detect, AFTER execution, when a widget's result
spanned tables with no validated safe relationship. Fail-closed throughout: any
missing provenance reads as UNSAFE (many-to-many), never safe.

Cardinality is taken from the edge's key_kind SHAPE via the canonical ingestion
mapping (`_relationship_type`); the sample-scoped card_a/card_b counts are NEVER
used to infer the class. The referential-validity floor is the same policy constant
the ingestion edge-creation gate uses — not a dashboard magic number.
"""
from __future__ import annotations

from app.services.semantic_layer_builder import _relationship_type
from app.services.semantic_policy import get_semantic_policy

# A unique ("one") side means a SUM on the fact side does not fan out. Only
# many-to-many carries the double-count risk.
_SAFE_CARDINALITIES = {"one_to_one", "one_to_many", "many_to_one"}


def classify_cardinality(edge: dict) -> str:
    """one_to_one | one_to_many | many_to_one | many_to_many. Fail-closed: any
    missing provenance -> many_to_many. card_a/card_b are NEVER used for the class
    (they are sample-scoped lower bounds; card_a == card_b does not imply 1:1)."""
    prov = (edge or {}).get("edge_provenance")
    if not prov:
        return "many_to_many"
    if prov.get("card_a") is None or prov.get("card_b") is None:
        return "many_to_many"
    ka, kb = prov.get("key_kind_a"), prov.get("key_kind_b")
    if ka is None or kb is None:
        return "many_to_many"
    return _relationship_type(ka, kb)


def safe_join(edge: dict) -> bool:
    """A join may be advertised / assumed safe iff its cardinality is not
    many-to-many AND its value overlap clears the ingestion referential floor.
    Fail-closed on a missing overlap."""
    overlap = (edge or {}).get("value_overlap_pct")
    if overlap is None:
        return False
    if overlap < get_semantic_policy().min_join_overlap:
        return False
    return classify_cardinality(edge) in _SAFE_CARDINALITIES


def _path_index(catalog) -> dict:
    """{blob_path | parquet_path | basename -> file_id} for resolving an agent's
    `files_used` (blob/parquet paths) back to catalog file_ids. Full paths are
    collision-free; a basename is only a usable key when it is UNAMBIGUOUS (maps to
    exactly one file_id) — a basename shared by two distinct tables is dropped so it
    can never confidently collapse two tables into one (fail-closed)."""
    idx: dict = {}
    basenames: dict = {}
    for t in catalog or []:
        fid = getattr(t, "file_id", None)
        if not fid:
            continue
        for p in (getattr(t, "blob_path", None), getattr(t, "parquet_path", None)):
            if p:
                idx[p] = fid  # full path: collision-free
                basenames.setdefault(str(p).rsplit("/", 1)[-1], set()).add(fid)
    for bn, fids in basenames.items():
        if len(fids) == 1 and bn not in idx:  # only unambiguous basenames resolve
            idx[bn] = next(iter(fids))
    return idx


def widget_join_safety(files_used, catalog, relationships) -> dict:
    """Post-execution honesty check. Maps the agent's result `files_used` (blob/
    parquet paths) -> catalog file_ids, then verifies that every spanned table pair
    is connected by a validated SAFE relationship. Returns
    {multi_table, safe, tables}. A single-table result is trivially safe."""
    idx = _path_index(catalog)
    ids: list = []
    for p in files_used or []:
        fid = idx.get(p) or idx.get(str(p).rsplit("/", 1)[-1])
        if fid and fid not in ids:
            ids.append(fid)
    if len(ids) <= 1:
        return {"multi_table": False, "safe": True, "tables": ids}
    safe_pairs = {
        frozenset((e.get("file_a_id"), e.get("file_b_id")))
        for e in (relationships or [])
        if safe_join(e)
    }
    ok = all(
        frozenset((a, b)) in safe_pairs
        for i, a in enumerate(ids)
        for b in ids[i + 1:]
    )
    return {"multi_table": True, "safe": ok, "tables": ids}
