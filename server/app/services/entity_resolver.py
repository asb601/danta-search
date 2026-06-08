"""Entity Resolution Engine — bridges BusinessIntentPlanner → retrieval.

RESPONSIBILITY (strictly bounded):
  Given normalized business entity names from BusinessIntentPlanner,
  find which catalog files are the best semantic candidates for each entity.

ONLY answers: WHERE do these entities physically live?

NOT responsible for:
  - SQL generation
  - join path resolution
  - graph traversal
  - retrieval execution
  - ontology expansion
  - business planning

Separation of concerns:
  PLANNER         → WHAT the user wants
  ENTITY RESOLVER → WHERE entities live  (this module)
  GRAPH           → HOW entities connect
  SQL GENERATOR   → HOW to execute

Design constraints:
  - No LLM. No recursive loops. One batch DB query per request.
  - Works against already-loaded catalog data (in-memory, per-request).
  - Scores are purely deterministic: metadata signals → weighted sum → cap.
  - Scales to millions of files: O(entities × catalog_size), linear.

Scoring signal hierarchy (strongest → weakest):
  1. semantic_role entity_key match  — this IS the master table for the entity
  2. semantic_role reference_key     — this REFERENCES the entity (transactions)
  3. key_dimensions overlap          — entity appears in filtering/grouping cols
    4. table/logical name ownership    — entity appears in the table name itself
    5. column_name overlap             — entity token in a raw column name
    6. ai_description coverage         — entity mentioned in the file description
    7. good_for coverage               — entity in natural-language use-case list
    8. key_metrics reference           — entity in metric column names (weak)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger
from app.models.file_metadata import FileMetadata


# ── Output types ───────────────────────────────────────────────────────────────

@dataclass
class EntityCandidate:
    """One candidate file/table for a resolved entity."""
    table: str          # blob_path — stable file identity used downstream
    file_id: str
    confidence: float   # [0.0, 1.0], rounded to 2 d.p.
    reason: str         # primary signal label (see _REASON_* constants)
    # Value-corroboration verdict for the HARD-pin gate. Set only when the pin
    # value-guard is active (RESOLVER_PIN_VALUE_GUARD_ENABLED). None = not
    # evaluated (legacy path). True = the entity's key role is carried by a
    # genuine key on this table per column_key_registry. Never affects confidence.
    corroborated: bool | None = None


@dataclass(frozen=True)
class RegistryKeyEvidence:
    """One column_key_registry row reduced to the fields the pin guard reads.

    The registry only ever stores genuine key columns (key_kind in {pk, fk});
    low-cardinality enums and non-key columns are not inserted. `semantic_role`
    is the same "custom:<kind>:<label>" string the resolver already parses.
    """
    column_name: str
    semantic_role: str | None
    key_kind: str
    cardinality: int


@dataclass
class EntityResolution:
    """Resolution result for one entity."""
    entity: str
    candidates: list[EntityCandidate] = field(default_factory=list)

    def best(self) -> EntityCandidate | None:
        return self.candidates[0] if self.candidates else None

    def to_dict(self) -> dict:
        return {
            "entity": self.entity,
            "candidates": [
                {
                    "table": c.table,
                    "file_id": c.file_id,
                    "confidence": c.confidence,
                    "reason": c.reason,
                }
                for c in self.candidates
            ],
        }


# ── Reason labels ──────────────────────────────────────────────────────────────

_REASON_ENTITY_KEY      = "semantic_role_match"     # entity_key role → master table
_REASON_REFERENCE_KEY   = "transactional_reference" # reference_key role → FK holder
_REASON_KEY_DIMENSION   = "key_dimension_match"
_REASON_TABLE_NAME      = "table_name_match"
_REASON_COLUMN_NAME     = "column_name_match"
_REASON_DESCRIPTION     = "description_match"
_REASON_GOOD_FOR        = "good_for_match"
_REASON_KEY_METRIC      = "metric_reference"


# ── Scoring weights ────────────────────────────────────────────────────────────
# Each weight is the maximum contribution when overlap = 1.0.
# Partial token overlap scales the contribution proportionally.

_W_ENTITY_KEY    = 0.55   # column is the primary key for this entity
_W_REFERENCE_KEY = 0.28   # column is a foreign-key reference to this entity
_W_KEY_DIMENSION = 0.38   # entity appears in key_dimensions
_W_TABLE_NAME    = 0.42   # entity appears in filename/logical table name
_W_COLUMN_NAME   = 0.28   # entity token in a raw column name
_W_DESCRIPTION   = 0.18   # entity mentioned in ai_description
_W_GOOD_FOR      = 0.14   # entity in good_for phrases
_W_KEY_METRIC    = 0.08   # entity in key_metrics (weakest — may be a measure)

_SCORE_CAP       = 0.95   # hard upper bound
_EMIT_THRESHOLD  = 0.14   # entries below this are noise; not emitted

# Minimum token overlap fraction required to activate a signal.
# This prevents spurious partial matches on short entities.
_MIN_OVERLAP     = 0.50


# ── Semantic role parsing ──────────────────────────────────────────────────────
# Role format: "custom:<kind>:<label>"  e.g. "custom:entity_key:profit_center"

_ROLE_RE = re.compile(r"^custom:([a-z_]+):(.+)$")
_HASH_PREFIX_RE = re.compile(r"^[0-9a-f]{8}_", re.IGNORECASE)


def _parse_role(role_str: str | None) -> tuple[str, str] | None:
    """Return (kind, label) if the role string is a valid dynamic role."""
    if not role_str:
        return None
    m = _ROLE_RE.match(str(role_str))
    return (m.group(1), m.group(2)) if m else None


def _extract_role_pairs(column_semantic_roles: dict | None) -> list[tuple[str, str]]:
    """
    Extract all (kind, label) pairs from a file's column_semantic_roles dict.

    Example input:  {"LIFNR": "custom:entity_key:vendor",
                     "BUKRS": "custom:reference_key:company_code"}
    Example output: [("entity_key", "vendor"), ("reference_key", "company_code")]
    """
    if not column_semantic_roles:
        return []
    pairs: list[tuple[str, str]] = []
    for role_str in column_semantic_roles.values():
        parsed = _parse_role(role_str)
        if parsed:
            pairs.append(parsed)
    return pairs


# ── Token normalization ────────────────────────────────────────────────────────

_NON_ALPHA_RE = re.compile(r"[^a-z0-9]+")


def _tokens(text: str) -> tuple[str, ...]:
    """Normalize text to ordered lowercase alphanumeric tokens (len ≥ 2)."""
    return tuple(dict.fromkeys(t for t in _NON_ALPHA_RE.split(text.lower()) if len(t) >= 2))


def _acronym(tokens: tuple[str, ...]) -> str:
    return "".join(t[0] for t in tokens if t)


def _clean_table_name(path: str) -> str:
    name = (path or "").rsplit("/", 1)[-1]
    name = _HASH_PREFIX_RE.sub("", name)
    return name.rsplit(".", 1)[0] if "." in name else name


def _table_overlap(entity_tokens: tuple[str, ...], target: str) -> float:
    """Strict ownership match for table names; partial phrases are not enough."""
    if not entity_tokens or not target:
        return 0.0
    target_tokens = _tokens(target)
    target_token_set = set(target_tokens)
    entity_acronym = _acronym(entity_tokens) if len(entity_tokens) > 1 else ""
    if entity_acronym and entity_acronym in target_token_set:
        return 1.0
    hits = sum(1 for t in entity_tokens if t in target_token_set)
    if len(entity_tokens) > 1 and hits < len(entity_tokens):
        return 0.0
    return hits / len(entity_tokens)


def _candidate_sort_name(path: str) -> str:
    return _clean_table_name(path).lower()


def _candidate_table_rank(path: str) -> int:
    tokens = set(_tokens(_candidate_sort_name(path)))
    if tokens & {"header", "headers", "master"}:
        return 0
    if tokens & {"line", "lines", "item", "items"}:
        return 1
    if tokens & {"distribution", "distributions", "schedule", "schedules"}:
        return 2
    return 3


def _overlap(entity_tokens: tuple[str, ...], target: str) -> float:
    """
    Fraction of entity tokens present in target text.

    Returns 0.0 if entity_tokens is empty or no match.
    Example: entity={"profit","center"}, target="Profit Center Code" → 1.0
             entity={"cost","center"}, target="Profit Center Code" → 0.5
    """
    if not entity_tokens or not target:
        return 0.0
    target_tokens = _tokens(target)
    target_token_set = set(target_tokens)
    entity_acronym = _acronym(entity_tokens) if len(entity_tokens) > 1 else ""
    target_acronym = _acronym(target_tokens) if len(target_tokens) > 1 else ""
    if entity_acronym and entity_acronym in target_token_set:
        return 1.0
    hits = sum(
        1 for t in entity_tokens
        if t in target_token_set or (2 <= len(t) <= 5 and t == target_acronym)
    )
    return hits / len(entity_tokens)


def _max_overlap(entity_tokens: tuple[str, ...], items: list[str]) -> float:
    """Max overlap of entity_tokens across a list of target strings."""
    if not items:
        return 0.0
    return max(_overlap(entity_tokens, item) for item in items)


# ── Per-entry scorer ───────────────────────────────────────────────────────────

def _score_entry(
    entity_tokens: tuple[str, ...],
    entry: dict,
    role_pairs: list[tuple[str, str]],   # (kind, label) for all columns in this file
) -> tuple[float, str]:
    """
    Score one catalog entry against the entity token set.

    Returns (confidence, primary_reason) where reason reflects the STRONGEST
    signal that contributed to the score.

    Additive scoring: each signal adds its weighted contribution independently.
    reason = label of the first (strongest) signal that fires.
    """
    score = 0.0
    reason = "no_match"
    table_text = " ".join(filter(None, [
        _clean_table_name(entry.get("blob_path") or ""),
        entry.get("logical_name") or "",
        entry.get("display_name") or "",
    ]))
    table_overlap = _table_overlap(entity_tokens, table_text)

    # ── 1. Semantic role — entity_key (master table) ───────────────────────
    # A file whose column carries entity_key:vendor IS the vendor master.
    entity_key_overlap = max(
        (_overlap(entity_tokens, label) for kind, label in role_pairs if kind == "entity_key"),
        default=0.0,
    )
    if entity_key_overlap >= _MIN_OVERLAP:
        score += _W_ENTITY_KEY * entity_key_overlap
        reason = _REASON_ENTITY_KEY

    # ── 2. Semantic role — reference_key (transactional reference) ─────────
    # A file whose column carries reference_key:vendor references the entity.
    ref_key_overlap = max(
        (_overlap(entity_tokens, label) for kind, label in role_pairs if kind == "reference_key"),
        default=0.0,
    )
    if ref_key_overlap >= _MIN_OVERLAP:
        score += _W_REFERENCE_KEY * ref_key_overlap
        if reason == "no_match":
            reason = _REASON_REFERENCE_KEY

    # ── 3. key_dimensions list ─────────────────────────────────────────────
    key_dims: list[str] = entry.get("key_dimensions") or []
    dim_overlap = _max_overlap(entity_tokens, key_dims)
    if dim_overlap >= _MIN_OVERLAP:
        score += _W_KEY_DIMENSION * dim_overlap
        if reason == "no_match":
            reason = _REASON_KEY_DIMENSION

    # ── 4. Column names ────────────────────────────────────────────────────
    if table_overlap >= _MIN_OVERLAP:
        score += _W_TABLE_NAME * table_overlap
        if reason == "no_match":
            reason = _REASON_TABLE_NAME

    # ── 5. Column names ────────────────────────────────────────────────────
    col_names: list[str] = entry.get("column_names") or []
    col_overlap = _max_overlap(entity_tokens, col_names)
    if col_overlap >= _MIN_OVERLAP:
        score += _W_COLUMN_NAME * col_overlap
        if reason == "no_match":
            reason = _REASON_COLUMN_NAME

    # ── 6. ai_description ─────────────────────────────────────────────────
    desc_overlap = _overlap(entity_tokens, entry.get("ai_description") or "")
    if desc_overlap >= _MIN_OVERLAP:
        score += _W_DESCRIPTION * desc_overlap
        if reason == "no_match":
            reason = _REASON_DESCRIPTION

    # ── 7. good_for phrases ────────────────────────────────────────────────
    good_for: list[str] = entry.get("good_for") or []
    gf_overlap = _max_overlap(entity_tokens, good_for)
    if gf_overlap >= _MIN_OVERLAP:
        score += _W_GOOD_FOR * gf_overlap
        if reason == "no_match":
            reason = _REASON_GOOD_FOR

    # ── 8. key_metrics (weakest) ───────────────────────────────────────────
    key_metrics: list[str] = entry.get("key_metrics") or []
    metric_overlap = _max_overlap(entity_tokens, key_metrics)
    if metric_overlap >= _MIN_OVERLAP:
        score += _W_KEY_METRIC * metric_overlap
        if reason == "no_match":
            reason = _REASON_KEY_METRIC

    # No structural anchor (no semantic role, no table name match): only weak
    # metadata signals fired. Generic attribute columns like 'status', 'type',
    # or 'code' appear in hundreds of unrelated tables — a dimension or column
    # hit without any ownership signal is unreliable. Cap to prevent false
    # positives from attribute bleed-through (e.g. MTL_SERIAL_NUMBERS scoring
    # high for 'po_current_status' because it has a STATUS key_dimension).
    if (entity_key_overlap < _MIN_OVERLAP
            and ref_key_overlap < _MIN_OVERLAP
            and table_overlap < _MIN_OVERLAP):
        score = min(score, 0.35)

    if entity_key_overlap >= _MIN_OVERLAP and table_overlap < _MIN_OVERLAP:
        score = min(score, 0.74)
        if reason == _REASON_ENTITY_KEY:
            reason = _REASON_REFERENCE_KEY

    return min(score, _SCORE_CAP), reason


# ── Value-corroboration gate on the HARD pin ─────────────────────────────────────
# This is NOT a re-scorer. Confidence scoring above is untouched. This gate only
# decides whether a would-be HARD pin (confidence >= resolver_pin_threshold) has
# real VALUE evidence behind its name/role-label match. A pin without a genuine
# key carrying the entity's role is a NAME-only guess and must not displace the
# correct (already-retrieved) table — nor silently break a P1 join whose other
# endpoint sits in the shortlist.

_REAL_KEY_KINDS = frozenset({"pk", "fk"})


def apply_pin_value_guard(
    entity_tokens: tuple[str, ...],
    registry_rows: list[RegistryKeyEvidence],
    min_join_cardinality: int,
) -> bool:
    """Is this entity's match on this table CORROBORATED BY VALUE EVIDENCE?

    True iff some column_key_registry row for the table is a genuine key —
        cardinality >= min_join_cardinality  AND  key_kind in {pk, fk}
    — whose semantic-role LABEL overlaps the entity (i.e. the entity's key role
    is actually carried by a real key on this table). No name heuristic: the
    overlap is the same role-label token overlap the resolver already uses.

    A low-cardinality enum, a non-key column, a key for a DIFFERENT entity, or
    a table absent from the registry → False (uncorroborated).
    """
    if not entity_tokens:
        return False
    for row in registry_rows:
        if row.key_kind not in _REAL_KEY_KINDS:
            continue
        if row.cardinality < min_join_cardinality:
            continue
        parsed = _parse_role(row.semantic_role)
        if not parsed:
            continue
        _kind, label = parsed
        if _overlap(entity_tokens, label) >= _MIN_OVERLAP:
            return True
    return False


def select_pinnable_blobs(
    resolution: dict[str, list[EntityCandidate]],
    pin_threshold: float,
    guard_enabled: bool,
) -> set[str]:
    """Decide which resolver candidates may be HARD-pinned into the shortlist.

    GATE on the existing pin promotion (no new scorer):

      guard OFF  → legacy: union of every candidate with confidence >= threshold.
      guard ON   → for each entity, among its candidates at/above the threshold,
                   keep only the value-CORROBORATED ones. If exactly one survives,
                   pin it. If 2+ survive (equally corroborated clones), ABSTAIN —
                   pin none for that entity (honest clone behavior, no name tie-break).
                   Uncorroborated would-be pins are dropped (they remain normal
                   shortlist candidates and compete on retrieval rank).
    """
    pinned: set[str] = set()
    for candidates in resolution.values():
        at_threshold = [c for c in candidates if c.confidence >= pin_threshold]
        if not guard_enabled:
            pinned.update(c.table for c in at_threshold)
            continue
        corroborated = [c for c in at_threshold if c.corroborated]
        if len(corroborated) == 1:
            pinned.add(corroborated[0].table)
        # len 0 → no value evidence → abstain; len 2+ → equal clones → abstain.
    return pinned


async def _annotate_pin_corroboration(
    result: dict[str, list[EntityCandidate]],
    db: AsyncSession,
) -> None:
    """Annotate each candidate's `corroborated` from column_key_registry.

    Flag-gated (RESOLVER_PIN_VALUE_GUARD_ENABLED) and FAIL-OPEN: any error leaves
    every `corroborated` as None, so select_pinnable_blobs (called with the same
    flag) falls back to legacy confidence-only pinning. ONE batch query, mirroring
    the role batch-fetch in resolve_entities — no per-candidate round-trips.
    """
    try:
        from app.core.config import get_settings  # noqa: PLC0415
        if not get_settings().RESOLVER_PIN_VALUE_GUARD_ENABLED:
            return
    except Exception:
        return

    candidate_file_ids = {
        c.file_id for cands in result.values() for c in cands if c.file_id
    }
    if not candidate_file_ids:
        return

    try:
        from app.models.column_key_registry import ColumnKeyRegistry  # noqa: PLC0415
        from app.services.semantic_policy import get_semantic_policy  # noqa: PLC0415

        min_card = get_semantic_policy().min_join_cardinality
        rows = (await db.execute(
            select(
                ColumnKeyRegistry.file_id,
                ColumnKeyRegistry.column_name,
                ColumnKeyRegistry.semantic_role,
                ColumnKeyRegistry.key_kind,
                ColumnKeyRegistry.cardinality,
            ).where(ColumnKeyRegistry.file_id.in_(candidate_file_ids))
        )).all()
    except Exception as exc:
        # Non-fatal: leave corroborated=None → guard fails open to legacy pinning.
        chat_logger.warning("entity_resolver_registry_fetch_error", error=str(exc)[:200])
        return

    evidence_map: dict[str, list[RegistryKeyEvidence]] = {}
    for row in rows:
        evidence_map.setdefault(row.file_id, []).append(RegistryKeyEvidence(
            column_name=row.column_name,
            semantic_role=row.semantic_role,
            key_kind=row.key_kind,
            cardinality=row.cardinality,
        ))

    for entity, cands in result.items():
        entity_tokens = _tokens(entity)
        for c in cands:
            c.corroborated = apply_pin_value_guard(
                entity_tokens, evidence_map.get(c.file_id, []), min_card
            )


# ── Public API ─────────────────────────────────────────────────────────────────

async def resolve_entities(
    entities: list[str],
    catalog: list[dict],
    db: AsyncSession,
    top_k: int = 3,
) -> dict[str, list[EntityCandidate]]:
    """
    Map each planner entity to its top-K candidate catalog files.

    Args:
        entities: normalized entity names from BusinessIntentPlan.entities
        catalog:  lean catalog records already loaded in _build_agent_context
        db:       async session — used for one batch semantic-roles fetch
        top_k:    max candidates per entity (default 3)

    Returns:
        { entity_name: [EntityCandidate, ...] } — sorted by confidence desc.
        Entities with no candidates above _EMIT_THRESHOLD get an empty list.

    DB cost: exactly ONE query (batch SELECT of column_semantic_roles for all
    visible files). No per-entity round-trips. No schema-wide scans.
    """
    result: dict[str, list[EntityCandidate]] = {e: [] for e in entities}

    if not entities or not catalog:
        return result

    # ── Step 1: Batch-fetch column_semantic_roles for all visible files ──────
    # column_semantic_roles is not in the lean catalog (too large to carry on
    # every request). We fetch it once here for all files, then index by file_id.
    file_ids = [e["file_id"] for e in catalog if e.get("file_id")]
    role_map: dict[str, list[tuple[str, str]]] = {}   # file_id → [(kind, label)]
    if file_ids:
        try:
            rows = (await db.execute(
                select(FileMetadata.file_id, FileMetadata.column_semantic_roles)
                .where(FileMetadata.file_id.in_(file_ids))
            )).all()
            for row in rows:
                role_map[row.file_id] = _extract_role_pairs(row.column_semantic_roles)
        except Exception as exc:
            # Non-fatal: continue without semantic roles; other signals still work
            chat_logger.warning("entity_resolver_role_fetch_error", error=str(exc)[:200])

    # ── Step 2: Score every entity × every catalog entry ────────────────────
    # Pure in-memory: O(entities × catalog_size). No more I/O after step 1.
    for entity in entities:
        entity_tokens = _tokens(entity)
        if not entity_tokens:
            continue

        candidates: list[EntityCandidate] = []
        for entry in catalog:
            file_id = entry.get("file_id") or ""
            blob_path = entry.get("blob_path") or file_id
            role_pairs = role_map.get(file_id, [])

            confidence, reason = _score_entry(entity_tokens, entry, role_pairs)

            if confidence >= _EMIT_THRESHOLD:
                candidates.append(EntityCandidate(
                    table=blob_path,
                    file_id=file_id,
                    confidence=round(confidence, 2),
                    reason=reason,
                ))

        reason_rank = {
            _REASON_ENTITY_KEY: 0,
            _REASON_TABLE_NAME: 1,
            _REASON_REFERENCE_KEY: 2,
            _REASON_KEY_DIMENSION: 3,
            _REASON_COLUMN_NAME: 4,
            _REASON_DESCRIPTION: 5,
            _REASON_GOOD_FOR: 6,
            _REASON_KEY_METRIC: 7,
        }
        candidates.sort(key=lambda c: (
            -c.confidence,
            reason_rank.get(c.reason, 99),
            _candidate_table_rank(c.table),
            _candidate_sort_name(c.table),
            c.table,
        ))
        deduped: list[EntityCandidate] = []
        seen_table_names: set[str] = set()
        for candidate in candidates:
            table_name = _candidate_sort_name(candidate.table)
            if table_name in seen_table_names:
                continue
            deduped.append(candidate)
            seen_table_names.add(table_name)
            if len(deduped) >= top_k:
                break
        result[entity] = deduped

    # ── Step 2.5: Value-corroboration for the HARD pin (flag-gated, fail-open) ─
    # Mirrors the role batch-fetch above: ONE batch SELECT of column_key_registry
    # for the candidate files, then a pure per-(entity, candidate) verdict. This
    # only annotates `corroborated`; it never reorders or rescoring candidates.
    # The graph pin gate (select_pinnable_blobs) reads this flag to decide whether
    # a would-be pin has real value evidence. Off → byte-identical legacy.
    await _annotate_pin_corroboration(result, db)

    # ── Step 3: Log summary ──────────────────────────────────────────────────
    chat_logger.info(
        "entity_resolution_done",
        entity_count=len(entities),
        resolved_count=sum(1 for v in result.values() if v),
        top_candidates={
            e: [(c.table.split("/")[-1], c.confidence, c.reason) for c in cands[:2]]
            for e, cands in result.items()
        },
    )

    return result


def resolution_to_dict(
    resolution: dict[str, list[EntityCandidate]],
) -> dict[str, list[dict]]:
    """Serialize entity resolution to a plain dict for ctx / logging."""
    return {
        entity: [
            {"table": c.table, "file_id": c.file_id, "confidence": c.confidence, "reason": c.reason}
            for c in candidates
        ]
        for entity, candidates in resolution.items()
    }
