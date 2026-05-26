"""Semantic Planner — sits between retrieval and LLM execution.

WHY THIS EXISTS
===============
Current flow (before planner):
  User query → retrieval → catalog → LangGraph agent → LLM generates SQL →
  DataFusion executes → synthesize answer

Problems with that flow:
  1. Every query pays LLM latency (1-3s) even for deterministic patterns.
  2. The LLM doesn't know which files are related — it guesses joins from raw column names.
  3. Runtime schema discovery: the agent explores schemas during the query. This is
     exactly what the ontology layer (ingestion-time role resolution) was built to prevent.
  4. LLM-generated JOIN conditions are often wrong when equivalent business keys
      have different raw column names across files.

New flow (with planner):
  User query → retrieval → catalog →
    [Semantic Planner, timeout=3s]
    → high confidence by policy: deterministic SQL → DataFusion → synthesize
      → low confidence: fall back to LangGraph agent (existing path)

HOW THE PLANNER WORKS
======================
1. Parse intent from user query (aggregation, dimension, time range, entities)
2. Load column_semantic_roles for candidate files (pre-computed at ingestion)
3. Match intent to typed roles: which file has the requested entity/dimension/measure?
4. Resolve join path: query FileRelationship WHERE semantic_role IN (shared_roles)
5. Generate deterministic SQL using role-typed column assignments
6. Compute confidence score — return plan only if it passes policy

WHAT v1 CAN HANDLE
==================
    ✓ Single-file aggregations over additive measures grouped by detected dimensions
    ✓ With time filter: "spend last month", "records in 2024"
    ✓ Single join when an approved SemanticRelationship exists
    ✓ Count queries grouped by dynamic entity/reference/attribute roles
    ✓ TOP-N queries over detected additive measures and dimensions
    ✓ Filter by entity/reference values when the relevant role is resolved

WHAT v1 FALLS BACK FOR
========================
  ✗ Multi-hop joins (depth > 2)
  ✗ Trend / period-over-period
  ✗ Comparison across two datasets
  ✗ Files with no roles resolved (no column_semantic_roles stored)
  ✗ Queries that need semantic reasoning beyond pattern matching
  ✗ Any timeout (hard 3s limit)

FALLBACK GUARANTEE
==================
If anything fails, times out, or confidence is too low, the planner returns
fallback_reason and the caller runs the existing LangGraph agent path. Zero risk.
Every fallback is logged with its reason to drive ontology coverage growth.
"""
from __future__ import annotations

import asyncio
import re
import time
from dataclasses import dataclass, field
from datetime import date, timedelta

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.logger import chat_logger
from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticRelationship
from app.services.file_identity import logical_name_from_path
from app.services.relationship_index import is_dictionary_like_path
from app.services.semantic_policy import get_semantic_policy
from app.services.semantic_roles import (
    dynamic_role_label,
    entity_name_for_role,
    is_additive_measure_role,
    is_date_role,
    is_metric_role,
    normalize_role_slug,
    role_kind,
    role_priority,
)

# ── Intent vocabulary ─────────────────────────────────────────────────────────

# Aggregation signals in the query
_AGG_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\btotal\b|\bsum\b|\badd up\b|\bcumulative\b", re.I), "SUM"),
    (re.compile(r"\baverage\b|\bavg\b|\bmean\b|\bper\s+unit\b", re.I), "AVG"),
    (re.compile(r"\bcount\b|\bhow many\b|\bnumber of\b|\bno\.\s+of\b|\b#\s+of\b", re.I), "COUNT"),
    (re.compile(r"\bmaximum\b|\bmax\b|\bhighest\b|\blargest\b|\bbiggest\b", re.I), "MAX"),
    (re.compile(r"\bminimum\b|\bmin\b|\blowest\b|\bsmallest\b", re.I), "MIN"),
]

# Dimension signals that do not depend on a business ontology.
_TIME_DIM_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bby month\b|\bmonthly\b|\bper month\b", re.I), "_month"),    # special
    (re.compile(r"\bby quarter\b|\bquarterly\b|\bper quarter\b", re.I), "_quarter"),  # special
    (re.compile(r"\bby year\b|\bannually\b|\byearly\b|\bper year\b", re.I), "_year"),  # special
]
_GROUP_BY_LABEL_RE = re.compile(
    r"\b(?:by|per|for each|group(?:ed)? by)\s+([a-zA-Z][a-zA-Z0-9_\- ]{1,48})",
    re.I,
)
_WISE_LABEL_RE = re.compile(r"\b([a-zA-Z][a-zA-Z0-9_\-]{1,48})[-\s]?wise\b", re.I)

# Time range signals
_TIME_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\blast\s+(\d+)\s+days?\b", re.I), "last_n_days"),
    (re.compile(r"\blast\s+(\d+)\s+months?\b", re.I), "last_n_months"),
    (re.compile(r"\blast\s+month\b", re.I), "last_month"),
    (re.compile(r"\blast\s+year\b|\bprevious\s+year\b", re.I), "last_year"),
    (re.compile(r"\bthis\s+month\b|\bcurrent\s+month\b", re.I), "this_month"),
    (re.compile(r"\bthis\s+year\b|\bcurrent\s+year\b", re.I), "this_year"),
    (re.compile(r"\bQ([1-4])\s*(\d{4})\b", re.I), "quarter"),
    (re.compile(
        r"\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|"
        r"jul(?:y)?|aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
        r"\s+(\d{4})\b", re.I,
    ), "month_year"),
    (re.compile(r"\b(20\d{2})\b"), "year"),
]

# Complexity signals — if matched, fall back to agent
_COMPLEX_PATTERNS = re.compile(
    r"\bcompare\b|\btrend\b|\bforecast\b|\bcorrelation\b|\bacross.+and\b"
    r"|\bversus\b|\bvs\.?\b|\bchange.+over\b|\bgrowth\b|\byoy\b|\bmom\b"
    r"|\brank.+against\b|\bbenchmark\b",
    re.I,
)

# Month name → number
_MONTH_MAP = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4,
    "may": 5, "jun": 6, "june": 6, "jul": 7, "july": 7,
    "aug": 8, "august": 8, "sep": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


# ── Data structures ────────────────────────────────────────────────────────────

@dataclass
class PlannedFile:
    file_id: str
    blob_path: str
    parquet_path: str | None
    alias: str                   # t0, t1, …
    primary_role: str            # strongest semantic role in this file
    roles: dict[str, str]        # col_name → semantic_role


@dataclass
class PlannedJoin:
    alias_a: str
    alias_b: str
    col_a: str
    col_b: str
    join_type: str               # INNER JOIN / LEFT JOIN
    role: str                    # semantic_role that makes this join valid
    confidence: float
    relationship_type: str       # one_to_one / one_to_many / many_to_one


@dataclass
class QueryIntent:
    aggregations: list[str]      # ["SUM"], ["COUNT"], ["RAW"] for detail queries
    group_by_roles: list[str]    # ["_label:<dimension>"], ["_month"], etc.
    time_filter: dict | None     # see _parse_time_filter()
    is_complex: bool
    top_n: int | None            # TOP-N queries
    raw_query: str


@dataclass
class ExecutionPlan:
    files: list[PlannedFile] = field(default_factory=list)
    joins: list[PlannedJoin] = field(default_factory=list)
    intent: QueryIntent | None = None
    sql: str | None = None
    confidence: float = 0.0
    fallback_reason: str | None = None
    planning_ms: float = 0.0


# ── Public API ─────────────────────────────────────────────────────────────────

async def plan(
    user_query: str,
    candidate_files: list[dict],
    db: AsyncSession,
    timeout_seconds: float = 3.0,
) -> ExecutionPlan:
    """Attempt a deterministic execution plan for the user query.

    Returns ExecutionPlan with:
    - sql set + confidence passes policy → caller should execute this SQL directly
      - fallback_reason set           → caller falls back to LangGraph agent

    Never raises. Always returns a valid ExecutionPlan.
    """
    t_start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            _plan_inner(user_query, candidate_files, db),
            timeout=timeout_seconds,
        )
        result.planning_ms = round((time.perf_counter() - t_start) * 1000, 2)
        chat_logger.info(
            "semantic_planner",
            confidence=round(result.confidence, 2),
            fallback_reason=result.fallback_reason,
            files_planned=len(result.files),
            joins_planned=len(result.joins),
            sql_generated=result.sql is not None,
            planning_ms=result.planning_ms,
            query_preview=user_query[:120],
        )
        return result
    except asyncio.TimeoutError:
        planning_ms = round((time.perf_counter() - t_start) * 1000, 2)
        chat_logger.warning(
            "semantic_planner_timeout",
            timeout_seconds=timeout_seconds,
            planning_ms=planning_ms,
            query_preview=user_query[:120],
        )
        return ExecutionPlan(fallback_reason="planner_timeout", planning_ms=planning_ms)
    except Exception as exc:
        planning_ms = round((time.perf_counter() - t_start) * 1000, 2)
        chat_logger.warning(
            "semantic_planner_error",
            error=str(exc)[:300],
            planning_ms=planning_ms,
            query_preview=user_query[:120],
        )
        return ExecutionPlan(
            fallback_reason=f"planner_error:{type(exc).__name__}",
            planning_ms=planning_ms,
        )


# ── Core planning logic ────────────────────────────────────────────────────────

async def _plan_inner(
    user_query: str,
    candidate_files: list[dict],
    db: AsyncSession,
) -> ExecutionPlan:
    result = ExecutionPlan()
    policy = get_semantic_policy()

    candidate_files = [
        f for f in candidate_files
        if not is_dictionary_like_path(f.get("blob_path") or f.get("name"))
    ]

    if not candidate_files:
        result.fallback_reason = "no_candidate_files"
        return result

    # ── 1. Parse intent ───────────────────────────────────────────────────────
    intent = _parse_intent(user_query)
    result.intent = intent

    if intent.is_complex:
        result.fallback_reason = "complex_query"
        return result

    # ── 2. Load semantic roles for candidate files ────────────────────────────
    candidate_ids = [f["file_id"] for f in candidate_files if f.get("file_id")]
    if not candidate_ids:
        result.fallback_reason = "no_file_ids"
        return result

    meta_rows = (await db.execute(
        select(
            FileMetadata.file_id,
            FileMetadata.blob_path,
            FileMetadata.column_semantic_roles,
        ).where(FileMetadata.file_id.in_(candidate_ids))
    )).all()

    # Build lookup: file_id → (blob_path, roles_dict)
    file_info: dict[str, tuple[str, dict[str, str]]] = {}
    for row in meta_rows:
        if row.column_semantic_roles:
            file_info[row.file_id] = (row.blob_path, row.column_semantic_roles)

    if not file_info:
        result.fallback_reason = "no_roles_in_candidate_files"
        return result

    # Parquet path lookup from catalog records
    parquet_map: dict[str, str | None] = {}
    for f in candidate_files:
        fid = f.get("file_id")
        if fid:
            parquet_map[fid] = f.get("parquet_blob_path") or f.get("blob_path")

    # ── 3. Assign files to planned roles ──────────────────────────────────────
    # Pick up to 4 files that have roles. Prioritise files with metric roles
    # since most analytical queries need those.
    ordered_ids = _rank_files_for_intent(file_info, intent, candidate_ids)
    planned_files: list[PlannedFile] = []
    for i, fid in enumerate(ordered_ids[:4]):
        blob, roles = file_info[fid]
        pf = PlannedFile(
            file_id=fid,
            blob_path=blob,
            parquet_path=parquet_map.get(fid),
            alias=f"t{i}",
            primary_role=_primary_role(roles),
            roles=roles,
        )
        planned_files.append(pf)

    if not planned_files:
        result.fallback_reason = "no_files_with_roles"
        return result

    result.files = planned_files

    # ── 4. Find approved semantic join paths ──────────────────────────────────
    if len(planned_files) >= 2:
        planned_ids = [f.file_id for f in planned_files]
        alias_map = {f.file_id: f.alias for f in planned_files}

        semantic_rows = (await db.execute(
            select(SemanticRelationship).where(
                SemanticRelationship.file_a_id.in_(planned_ids),
                SemanticRelationship.file_b_id.in_(planned_ids),
                SemanticRelationship.status == "active",
                SemanticRelationship.confidence_score >= policy.planner_join_min_confidence,
            ).order_by(SemanticRelationship.confidence_score.desc())
        )).scalars().all()

        seen_pairs: set[frozenset] = set()
        candidate_reasons: list[str] = []
        for rel in semantic_rows:
            pair = frozenset([rel.file_a_id, rel.file_b_id])
            if pair in seen_pairs:
                continue

            if rel.approval_status != "approved":
                if rel.risk_reason:
                    candidate_reasons.append(rel.risk_reason)
                continue

            seen_pairs.add(pair)

            role = (rel.join_rule or {}).get("semantic_role") or "approved_join"
            if not rel.from_column or not rel.to_column:
                continue

            result.joins.append(PlannedJoin(
                alias_a=alias_map[rel.file_a_id],
                alias_b=alias_map[rel.file_b_id],
                col_a=rel.from_column,
                col_b=rel.to_column,
                join_type=(rel.join_rule or {}).get("join_type") or "LEFT JOIN",
                role=role,
                confidence=rel.confidence_score,
                relationship_type=rel.relationship_type,
            ))

        if semantic_rows and not result.joins and candidate_reasons:
            result.fallback_reason = "candidate_semantic_relationship:" + candidate_reasons[0][:120]
            result.confidence = 0.0
            return result

    # ── 5. Compute confidence ─────────────────────────────────────────────────
    result.confidence = _compute_confidence(result, intent)

    # ── 6. Generate SQL if confidence is sufficient ───────────────────────────
    if result.confidence >= policy.planner_fast_path_confidence:
        sql = _generate_sql(result, intent)
        if sql:
            result.sql = sql
        else:
            result.fallback_reason = "sql_generation_failed"
            result.confidence = 0.0
    else:
        result.fallback_reason = f"low_confidence:{round(result.confidence, 2)}"

    return result


# ── Intent parsing ─────────────────────────────────────────────────────────────

def _clean_group_label(value: str) -> str | None:
    cleaned = re.split(
        r"\b(?:where|when|for|from|in|on|during|last|this|with|and|order|sort|limit|top)\b",
        value.strip(),
        maxsplit=1,
        flags=re.I,
    )[0]
    slug = normalize_role_slug(cleaned)
    return slug if slug and slug not in {"month", "quarter", "year"} else None


def _parse_intent(query: str) -> QueryIntent:
    q = query

    # Complexity check first — bail early
    is_complex = bool(_COMPLEX_PATTERNS.search(q))

    # Aggregation
    aggs: list[str] = []
    for pat, agg in _AGG_PATTERNS:
        if pat.search(q):
            aggs.append(agg)

    # TOP-N
    top_n: int | None = None
    top_m = re.search(r"\btop\s+(\d+)\b", q, re.I)
    if top_m:
        top_n = int(top_m.group(1))
        if "MAX" not in aggs:
            aggs.append("SUM")

    if not aggs:
        # Detail / list queries
        if re.search(r"\bshow\b|\blist\b|\bget\b|\bfetch\b|\bdisplay\b|\ball\b", q, re.I):
            aggs = ["RAW"]
        else:
            # Default to SUM for analytical queries (most common)
            aggs = ["SUM"]

    # Dimension (GROUP BY)
    group_by_roles: list[str] = []
    for pat, role in _TIME_DIM_PATTERNS:
        if pat.search(q):
            group_by_roles.append(role)

    for match in _GROUP_BY_LABEL_RE.finditer(q):
        label = _clean_group_label(match.group(1))
        if label:
            group_by_roles.append(f"_label:{label}")

    for match in _WISE_LABEL_RE.finditer(q):
        label = _clean_group_label(match.group(1))
        if label:
            group_by_roles.append(f"_label:{label}")

    group_by_roles = list(dict.fromkeys(group_by_roles))

    # Time filter
    time_filter = _parse_time_filter(q)

    return QueryIntent(
        aggregations=aggs,
        group_by_roles=group_by_roles,
        time_filter=time_filter,
        is_complex=is_complex,
        top_n=top_n,
        raw_query=query,
    )


def _parse_time_filter(query: str) -> dict | None:
    """Extract time range from user query. Returns a dict describing the filter."""
    today = date.today()

    for pat, ttype in _TIME_PATTERNS:
        m = pat.search(query)
        if not m:
            continue

        if ttype == "last_n_days":
            n = int(m.group(1))
            start = today - timedelta(days=n)
            return {"type": "range", "start": start.isoformat(), "end": today.isoformat()}

        if ttype == "last_n_months":
            n = int(m.group(1))
            start = today.replace(day=1)
            for _ in range(n):
                start = (start - timedelta(days=1)).replace(day=1)
            return {"type": "range", "start": start.isoformat(), "end": today.isoformat()}

        if ttype == "last_month":
            first_this = today.replace(day=1)
            last_last = first_this - timedelta(days=1)
            first_last = last_last.replace(day=1)
            return {"type": "range", "start": first_last.isoformat(),
                    "end": last_last.isoformat()}

        if ttype == "last_year":
            y = today.year - 1
            return {"type": "year", "year": y}

        if ttype == "this_month":
            first = today.replace(day=1)
            return {"type": "range", "start": first.isoformat(), "end": today.isoformat()}

        if ttype == "this_year":
            return {"type": "year", "year": today.year}

        if ttype == "quarter":
            q_name = m.group(1).upper()
            year = int(m.group(2))
            q_num = int(q_name)  # group(1) captures the digit: "1", "2", "3", "4"
            q_start_month = (q_num - 1) * 3 + 1
            q_start = date(year, q_start_month, 1)
            q_end_month = q_start_month + 2
            import calendar
            q_end_day = calendar.monthrange(year, q_end_month)[1]
            q_end = date(year, q_end_month, q_end_day)
            return {"type": "range", "start": q_start.isoformat(),
                    "end": q_end.isoformat()}

        if ttype == "month_year":
            month_str = m.group(1).lower()[:3]
            year = int(m.group(2))
            month_num = _MONTH_MAP.get(month_str)
            if month_num:
                import calendar
                last_day = calendar.monthrange(year, month_num)[1]
                return {
                    "type": "range",
                    "start": date(year, month_num, 1).isoformat(),
                    "end": date(year, month_num, last_day).isoformat(),
                }

        if ttype == "year":
            year = int(m.group(1))
            if 2000 <= year <= 2035:
                return {"type": "year", "year": year}

    return None


# ── File ranking and role utilities ───────────────────────────────────────────

def _rank_files_for_intent(
    file_info: dict[str, tuple[str, dict[str, str]]],
    intent: QueryIntent,
    candidate_order: list[str],
) -> list[str]:
    """Order file_ids so the most relevant file (for this intent) comes first."""
    def _score(fid: str) -> int:
        _, roles = file_info[fid]
        role_set = set(roles.values())
        score = 0
        # Files with metric roles are always needed for aggregation
        if intent.aggregations != ["RAW"]:
            if any(is_metric_role(role) for role in role_set):
                score += 10
        # Files that match a requested dimension
        for grole in intent.group_by_roles:
            if grole.startswith("_"):
                if grole.startswith("_label:"):
                    label = grole.split(":", 1)[1]
                    if _col_for_label(roles, label):
                        score += 5
                continue  # time-derived dimension — any file with a date role qualifies
            if grole in role_set:
                score += 5
        # Files with date columns are valuable when there's a time filter
        if intent.time_filter and any(is_date_role(role) for role in role_set):
            score += 3
        return score

    # Preserve candidate_order as tiebreaker (retrieval already ranked these)
    ids_with_roles = [fid for fid in candidate_order if fid in file_info]
    return sorted(ids_with_roles, key=_score, reverse=True)


def _primary_role(roles: dict[str, str]) -> str:
    """Return the most significant semantic role in a file."""
    role_set = set(roles.values())
    if not role_set:
        return "unknown"
    return min(role_set, key=role_priority)


def _col_for_role(roles: dict[str, str], target_role: str) -> str | None:
    """Return the first column name that carries target_role."""
    for col, role in roles.items():
        if role == target_role:
            return col
    return None


def _role_matches_label(role: str, label: str) -> bool:
    label_slug = normalize_role_slug(label)
    candidates = {normalize_role_slug(role)}
    for value in (dynamic_role_label(role), entity_name_for_role(role)):
        if value:
            candidates.add(normalize_role_slug(value))
    return label_slug in candidates or any(label_slug and label_slug in candidate for candidate in candidates)


def _col_for_label(roles: dict[str, str], label: str) -> str | None:
    ranked = sorted(roles.items(), key=lambda item: role_priority(item[1]))
    label_slug = normalize_role_slug(label)
    for col, role in ranked:
        if _role_matches_label(role, label_slug) or label_slug in normalize_role_slug(col):
            return col
    return None


def _col_for_kind(roles: dict[str, str], target_kind: str) -> str | None:
    """Return the first column name that carries a role of target_kind."""
    ranked = sorted(roles.items(), key=lambda item: role_priority(item[1]))
    for col, role in ranked:
        if role_kind(role) == target_kind:
            return col
    return None


def _metric_roles_for_aggregation(roles: dict[str, str], aggregation: str) -> list[str]:
    metric_roles = sorted(
        {role for role in roles.values() if is_metric_role(role)},
        key=role_priority,
    )
    if aggregation == "SUM":
        return [role for role in metric_roles if is_additive_measure_role(role)]
    return metric_roles


# ── Confidence scoring ─────────────────────────────────────────────────────────

def _compute_confidence(result: ExecutionPlan, intent: QueryIntent) -> float:
    policy = get_semantic_policy()
    score = 0.0

    if not result.files:
        return 0.0

    primary = result.files[0]
    role_set = set(primary.roles.values())

    # Primary file has metric column → can aggregate
    has_metric = any(is_metric_role(role) for role in role_set)
    if has_metric and intent.aggregations != ["RAW"]:
        score += policy.planner_metric_bonus
    elif intent.aggregations == ["RAW"]:
        score += policy.planner_raw_intent_bonus

    # Aggregation intent is clear
    if intent.aggregations and intent.aggregations != ["SUM"]:
        # Explicit aggregation keyword → higher confidence
        score += policy.planner_explicit_aggregation_bonus
    else:
        score += policy.planner_default_aggregation_bonus

    # Dimension (GROUP BY) intent identified
    if intent.group_by_roles:
        # Check if at least one requested dimension role exists in files
        all_roles = set()
        for pf in result.files:
            all_roles |= set(pf.roles.values())
        matched_dims = sum(
            1 for r in intent.group_by_roles
            if r in all_roles
            or r in {"_month", "_quarter", "_year"}
            or (
                r.startswith("_label:")
                and any(_col_for_label(pf.roles, r.split(":", 1)[1]) for pf in result.files)
            )
        )
        if matched_dims:
            score += policy.planner_dimension_bonus

    # Time filter resolved
    if intent.time_filter:
        # Check if any file has an event_date column
        has_date_col = any(is_date_role(role) for pf in result.files for role in pf.roles.values())
        if has_date_col:
            score += policy.planner_time_filter_bonus
        else:
            score -= policy.planner_missing_time_filter_penalty

    # Valid join path found (for multi-file queries)
    if len(result.files) >= 2 and result.joins:
        score += policy.planner_join_bonus
    elif len(result.files) >= 2 and not result.joins:
        # Multiple files but no relationship — likely need agent to figure it out
        score -= policy.planner_missing_join_penalty

    # Penalise single file with default SUM and no dimension — too vague
    if (len(result.files) == 1 and intent.aggregations == ["SUM"]
            and not intent.group_by_roles and not intent.time_filter):
        score -= policy.planner_vague_single_file_penalty

    return min(max(score, 0.0), 1.0)


def _generate_sql(result: ExecutionPlan, intent: QueryIntent) -> str | None:
    """Generate logical SQL from the execution plan.

    The caller canonicalizes logical table names to physical storage through
    FileIdentityMap before execution. The planner must not emit blob paths.

    Returns None if a required piece (e.g. metric column) is missing.
    """
    if not result.files:
        return None

    primary = result.files[0]
    agg = intent.aggregations[0] if intent.aggregations else "SUM"
    is_raw = agg == "RAW"

    # ── FROM clause ──────────────────────────────────────────────────────────
    primary_table = logical_name_from_path(primary.blob_path)
    if not primary_table:
        return None

    primary_src = primary_table

    # ── SELECT clause ─────────────────────────────────────────────────────────
    select_parts: list[str] = []
    group_by_exprs: list[str] = []

    # Dimension columns
    for grole in intent.group_by_roles:
        if grole == "_month":
            # Find date column in primary file
            date_col = _col_for_role(primary.roles, "event_date") or _col_for_kind(primary.roles, "date")
            if date_col:
                expr = f"DATE_TRUNC('month', TRY_CAST({_q(date_col)} AS TIMESTAMP)) AS period"
                select_parts.append(expr)
                group_by_exprs.append(f"DATE_TRUNC('month', TRY_CAST({_q(date_col)} AS TIMESTAMP))")
        elif grole == "_quarter":
            date_col = _col_for_role(primary.roles, "event_date") or _col_for_kind(primary.roles, "date")
            if date_col:
                expr = (
                    f"CONCAT(CAST(EXTRACT(YEAR FROM TRY_CAST({_q(date_col)} AS DATE)) AS VARCHAR), "
                    f"'-Q', CAST(CEIL(EXTRACT(MONTH FROM TRY_CAST({_q(date_col)} AS DATE)) / 3.0) AS VARCHAR)) AS period"
                )
                select_parts.append(expr)
                group_by_exprs.append(
                    f"CONCAT(CAST(EXTRACT(YEAR FROM TRY_CAST({_q(date_col)} AS DATE)) AS VARCHAR), "
                    f"'-Q', CAST(CEIL(EXTRACT(MONTH FROM TRY_CAST({_q(date_col)} AS DATE)) / 3.0) AS VARCHAR))"
                )
        elif grole == "_year":
            date_col = _col_for_role(primary.roles, "event_date") or _col_for_kind(primary.roles, "date")
            if date_col:
                expr = f"EXTRACT(YEAR FROM TRY_CAST({_q(date_col)} AS DATE)) AS year"
                select_parts.append(expr)
                group_by_exprs.append(f"EXTRACT(YEAR FROM TRY_CAST({_q(date_col)} AS DATE))")
        elif grole.startswith("_label:"):
            label = grole.split(":", 1)[1]
            dim_col = _col_for_label(primary.roles, label)
            if dim_col:
                select_parts.append(f"t0.{_q(dim_col)}")
                group_by_exprs.append(f"t0.{_q(dim_col)}")
            else:
                for j in result.joins:
                    other = _find_file_by_alias(result.files, j.alias_b if j.alias_a == "t0" else j.alias_a)
                    if other:
                        dim_col = _col_for_label(other.roles, label)
                        if dim_col:
                            alias = other.alias
                            select_parts.append(f"{alias}.{_q(dim_col)}")
                            group_by_exprs.append(f"{alias}.{_q(dim_col)}")
                            break
        else:
            # Regular dimension column — find it in primary file or joined files
            dim_col = _col_for_role(primary.roles, grole)
            if dim_col:
                select_parts.append(f"t0.{_q(dim_col)}")
                group_by_exprs.append(f"t0.{_q(dim_col)}")
            else:
                # Look in joined files
                for j in result.joins:
                    other = _find_file_by_alias(result.files, j.alias_b if j.alias_a == "t0" else j.alias_a)
                    if other:
                        dim_col = _col_for_role(other.roles, grole)
                        if dim_col:
                            alias = other.alias
                            select_parts.append(f"{alias}.{_q(dim_col)}")
                            group_by_exprs.append(f"{alias}.{_q(dim_col)}")
                            break

    # Metric column(s)
    metric_col: str | None = None
    metric_label: str | None = None
    if not is_raw:
        for role in _metric_roles_for_aggregation(primary.roles, agg):
            metric_col = _col_for_role(primary.roles, role)
            if metric_col:
                metric_label = f"{agg.lower()}_{metric_col.lower()}"
                if agg == "COUNT":
                    select_parts.append(f"COUNT(*) AS {_q(metric_label)}")
                else:
                    select_parts.append(
                        f"{agg}(TRY_CAST(t0.{_q(metric_col)} AS DOUBLE)) AS {_q(metric_label)}"
                    )
                break

        if not metric_col and agg != "COUNT":
            # No monetary column found — can't aggregate meaningfully
            return None

    # If COUNT and no dimension, just count records
    if agg == "COUNT" and not metric_col and not select_parts:
        select_parts.append("COUNT(*) AS record_count")

    # Raw detail query — select all columns from primary
    if is_raw:
        select_parts = ["t0.*"]
        group_by_exprs = []

    if not select_parts:
        return None

    # ── JOIN clauses ──────────────────────────────────────────────────────────
    join_clauses: list[str] = []
    from_parts = f"{primary_src} AS t0"

    for j in result.joins:
        if j.alias_a == "t0":
            other_alias = j.alias_b
            primary_col = j.col_a
            other_col = j.col_b
        elif j.alias_b == "t0":
            other_alias = j.alias_a
            primary_col = j.col_b
            other_col = j.col_a
        else:
            continue

        other = _find_file_by_alias(result.files, other_alias)
        if not other:
            continue
        other_src = logical_name_from_path(other.blob_path)
        if not other_src:
            continue
        join_clauses.append(
            f"{j.join_type} {other_src} AS {other.alias}\n"
            f"  ON t0.{_q(primary_col)} = {other.alias}.{_q(other_col)}"
        )

    # ── WHERE clause ──────────────────────────────────────────────────────────
    where_parts: list[str] = []
    if intent.time_filter:
        date_col = _col_for_role(primary.roles, "event_date")
        if date_col:
            tf = intent.time_filter
            if tf["type"] == "range":
                where_parts.append(
                    f"TRY_CAST(t0.{_q(date_col)} AS DATE) BETWEEN DATE '{tf['start']}' AND DATE '{tf['end']}'"
                )
            elif tf["type"] == "year":
                where_parts.append(
                    f"EXTRACT(YEAR FROM TRY_CAST(t0.{_q(date_col)} AS DATE)) = {tf['year']}"
                )

    # ── Assemble SQL ──────────────────────────────────────────────────────────
    sql = f"SELECT {', '.join(select_parts)}\nFROM {from_parts}"
    for jc in join_clauses:
        sql += f"\n{jc}"
    if where_parts:
        sql += f"\nWHERE {' AND '.join(where_parts)}"
    if group_by_exprs and not is_raw:
        sql += f"\nGROUP BY {', '.join(group_by_exprs)}"
    if not is_raw and metric_label:
        if intent.top_n:
            sql += f"\nORDER BY {_q(metric_label)} DESC\nLIMIT {intent.top_n}"
        elif group_by_exprs:
            sql += f"\nORDER BY {_q(metric_label)} DESC\nLIMIT 50"
        else:
            sql += "\nLIMIT 50"
    elif is_raw:
        sql += "\nLIMIT 100"

    return sql


# ── Helpers ────────────────────────────────────────────────────────────────────

def _q(name: str) -> str:
    """Quote a column or label name for DataFusion SQL."""
    # Use double-quotes for identifiers — DataFusion follows ANSI SQL
    return f'"{name}"'


def _find_file_by_alias(files: list[PlannedFile], alias: str) -> PlannedFile | None:
    for f in files:
        if f.alias == alias:
            return f
    return None
