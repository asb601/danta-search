"""Bounded SQL repair — deterministic pattern fixes + focused LLM fallback.

HARD LIMITS (not configurable):
  MAX_ATTEMPTS = 2  per run_sql invocation.
  LLM repair output capped at 512 tokens.
  No recursion. No replanning. No new retrieval. No new context.

Repair tiers:
  Tier 1 — deterministic, zero-cost, <1ms
    Only applies fixes that are mechanically safe and structurally bounded.
    Explicitly does NOT attempt semantic interpretation.

  Tier 2 — focused LLM call (only when Tier 1 makes no change)
    Sends: error + failing SQL + approved joins/columns (compact).
    Receives: corrected SQL only — validated for intent preservation before use.
    Uses same sync Azure OpenAI client as the rest of the service.

DESIGN POLICY — what belongs in each tier:

  Deterministic (Tier 1) is appropriate for:
    - Finite mechanical rewrites with no semantic ambiguity
    - Execution-level syntax normalization (CAST → TRY_CAST)
    - NULL semantics where the column role is explicitly catalogued
    - Structural fixes with confirmable preconditions (GROUP BY with SELECT check)

  LLM (Tier 2) is appropriate for:
    - Open-vocabulary language problems: alias mismatch, column name drift
    - Dialect-specific syntax that requires language understanding
    - Any repair requiring semantic interpretation of business vocabulary

  NEVER appropriate (neither tier):
    - Growing synonym / alias dictionaries for business vocabulary
    - Heuristic keyword expansion (vendor = supplier = payee = ...)
    - Semantic interpretation of what a column "probably means"
    These belong in the EntityResolver + BusinessIntentPlanner, not here.

INVARIANTS (never violated):
  - Never adds tables/files not already present in the SQL.
  - Never changes JOIN columns to anything outside approved_joins.
  - Never alters GROUP BY presence/absence (analytical granularity).
  - Never introduces SELECT * if original didn't have it.
  - Returns None when no repair is possible — caller treats as final failure.
  - Repaired SQL re-validated through validate_and_normalise before use.

SQL transformation note:
  Tier 1 rewrites use regex surgery. This is acceptable for the narrow,
  well-understood patterns implemented here (CAST normalization, = '' guard).
  Migration path: when sqlglot is added as a dependency, replace these with
  AST-level transforms (sqlglot.transpile / expression tree mutations) for
  structural safety against nested expressions and string literals.

Typical call pattern (from run_sql):
  for attempt in range(MAX_REPAIR + 1):
      try: execute(sql)
      except DBError as exc:
          repaired = attempt_repair(sql, exc, sql_ctx, attempt)
          if repaired: sql = repaired; continue
          break
"""
from __future__ import annotations

import re
import time

from app.core.logger import chat_logger, pipeline_logger
from app.core.openai_client import get_client
from app.policies.repair_policy import get_repair_policy as _get_repair_policy
import app.services.sql_ast as _sql_ast

# ── Tier 1: deterministic pattern matchers ─────────────────────────────────────

# CAST(expr AS TYPE) — captured groups: (1) expr, (2) type name
# NOTE: regex CAST rewriting is safe for non-nested CAST expressions only.
# The guard _has_nested_cast() must pass before applying this pattern.
# Migration path: replace with sqlglot AST transform once sqlglot is a dependency.
_CAST_RE = re.compile(
    r"\bCAST\s*\((.+?)\s+AS\s+(DATE|TIMESTAMP|INTEGER|INT|INT32|INT64|BIGINT|FLOAT|DOUBLE|DECIMAL)\b",
    re.IGNORECASE,
)

# column = '' or column = "" — for reference_key null-semantics columns only
_EMPTY_EQ_RE = re.compile(r"([\w.]+)\s*=\s*(?:''|\"\")")

# DuckDB GROUP BY binder error: column "X" must appear in the GROUP BY clause
_GROUPBY_MISSING_RE = re.compile(
    r'column\s+"([^"]+)"\s+must appear in the GROUP BY', re.IGNORECASE
)

# GROUP BY clause body — captures everything after GROUP BY until ORDER/HAVING/LIMIT/end
_GROUPBY_CLAUSE_RE = re.compile(
    r"\bGROUP\s+BY\s+([\s\S]+?)(?=\s+(?:HAVING|ORDER|LIMIT|UNION|EXCEPT|INTERSECT)\b|;|\Z)",
    re.IGNORECASE,
)

# SELECT clause body — everything between SELECT [DISTINCT] and FROM
_SELECT_CLAUSE_RE = re.compile(
    r"\bSELECT\s+(?:DISTINCT\s+)?([\s\S]+?)\s+FROM\b",
    re.IGNORECASE,
)

# Aggregate function patterns — used to strip aggregate bodies from SELECT analysis
_AGG_BODY_RE = re.compile(
    r"\b(SUM|COUNT|AVG|MIN|MAX|STDDEV|VARIANCE|ARRAY_AGG|STRING_AGG|GROUP_CONCAT)"
    r"\s*\([^()]*\)",
    re.IGNORECASE,
)

# For intent validation — extract az:// file references
_AZ_PATH_RE = re.compile(r"az://[^\s'\"]+", re.IGNORECASE)

# Error category fingerprints (match against str(exc).lower())
_ERR_CONVERSION = ("conversion error", "could not convert", "invalid cast",
                   "failed to cast", "cannot cast")
_ERR_GROUPBY = ("must appear in the group by",)


# ── Structural guards ──────────────────────────────────────────────────────────

def _has_nested_cast(sql: str) -> bool:
    """
    Return True if the SQL contains nested CAST expressions.

    Uses bracket counting rather than regex to reliably detect nesting.
    Nested CASTs like CAST(CAST(x AS INT) AS DOUBLE) defeat the flat CAST_RE
    pattern and would produce malformed output — we decline and let Tier 2 handle.
    """
    cast_positions = [m.end() for m in re.finditer(r"\bCAST\s*\(", sql, re.IGNORECASE)]
    if len(cast_positions) < 2:
        return False  # zero or one CAST — nesting not possible
    for start in cast_positions:
        depth, pos = 1, start
        while pos < len(sql) and depth > 0:
            if sql[pos] == "(":
                depth += 1
            elif sql[pos] == ")":
                depth -= 1
            pos += 1
        inner = sql[start : pos - 1]
        if re.search(r"\bCAST\b", inner, re.IGNORECASE):
            return True
    return False


def _column_is_bare_in_select(sql: str, column: str) -> bool:
    """
    Return True if `column` appears in the SELECT clause as a non-aggregated token.

    This is the precondition for the GROUP BY auto-append fix:
    we only add to GROUP BY when we can confirm the column is a grouping dimension,
    not a metric that happens to share the column name.

    Approach:
      1. Extract SELECT clause (between SELECT and FROM).
      2. Remove aggregate function bodies (SUM(...), COUNT(...), etc.).
      3. Check if `column` appears as a word in the remaining text.
    """
    m = _SELECT_CLAUSE_RE.search(sql)
    if not m:
        return False
    select_clause = m.group(1)
    # Strip aggregate function bodies so we only see bare dimension columns
    stripped = _AGG_BODY_RE.sub("__agg__()", select_clause)
    return bool(re.search(r"\b" + re.escape(column) + r"\b", stripped, re.IGNORECASE))


# ── Intent validation: LLM repair must not change analytical structure ─────────

_HAS_GROUPBY_RE = re.compile(r"\bGROUP\s+BY\b", re.IGNORECASE)
_HAS_SELECT_STAR_RE = re.compile(r"\bSELECT\s+\*", re.IGNORECASE)

# Aggregate function names — used to count aggregate expressions in SELECT
_AGG_FUNC_RE = re.compile(
    r"\b(SUM|COUNT|AVG|MIN|MAX|STDDEV|VARIANCE|ARRAY_AGG|STRING_AGG|GROUP_CONCAT)\s*\(",
    re.IGNORECASE,
)

# GROUP BY clause — capture clause body (same as _GROUPBY_CLAUSE_RE but independent copy)
_GB_BODY_RE = re.compile(
    r"\bGROUP\s+BY\s+([\s\S]+?)(?=\s+(?:HAVING|ORDER|LIMIT|UNION|EXCEPT|INTERSECT)\b|;|\Z)",
    re.IGNORECASE,
)


def _count_aggregates_in_select(sql: str) -> int:
    """
    Count aggregate function calls in the SELECT clause only.

    Extracts the SELECT clause (between SELECT and FROM), strips any
    window function OVER(...) bodies to avoid double-counting, then counts
    aggregate function matches. Returns 0 if SELECT clause cannot be extracted.
    """
    m = _SELECT_CLAUSE_RE.search(sql)
    if not m:
        return 0
    select_clause = m.group(1)
    # Strip OVER (...) window specs so we don't confuse window agg with plain agg
    stripped = re.sub(r"\bOVER\s*\([^)]*\)", "", select_clause, flags=re.IGNORECASE)
    return len(_AGG_FUNC_RE.findall(stripped))


def _extract_groupby_tokens(sql: str) -> frozenset[str]:
    """
    Extract normalised (uppercase, stripped) token set from the GROUP BY clause.

    Returns empty frozenset if no GROUP BY present.
    Used to detect column removal from GROUP BY — removing a dimension changes
    analytical granularity just as much as adding one.
    """
    m = _GB_BODY_RE.search(sql)
    if not m:
        return frozenset()
    # Split on comma, strip aliases and whitespace, upper-case
    raw = m.group(1)
    tokens: set[str] = set()
    for part in raw.split(","):
        # Drop ORDER BY / HAVING trailer tokens that sneak in at the end
        clean = re.split(r"\b(HAVING|ORDER|LIMIT)\b", part, flags=re.IGNORECASE)[0]
        token = clean.strip().upper()
        if token:
            tokens.add(token)
    return frozenset(tokens)


def _validate_repair_intent(original: str, repaired: str) -> bool:
    """
    Return True only when the repair preserved all structural analytical constraints.

    Checks (rejection on any failure):
    1. No new az:// file references introduced (no scope expansion).
    2. GROUP BY presence unchanged — both have it, or neither does.
    3. GROUP BY columns not removed — repairing SQL must not drop dimensions.
       Removing a GROUP BY column collapses analytical granularity silently.
    4. Aggregate count in SELECT unchanged — repair must not remove or add
       aggregation logic. ± 0 required; any delta = structural change.
    5. SELECT * not introduced if original didn't use it.

    Each check is independent — all five must pass.
    Intentionally structural-only: no semantic comparison, no LLM reasoning.
    """
    # ── Check 1: No new file scope ─────────────────────────────────────────────
    orig_paths = set(_AZ_PATH_RE.findall(original))
    rep_paths = set(_AZ_PATH_RE.findall(repaired))
    if rep_paths - orig_paths:
        chat_logger.warning(
            "sql_repair_intent_new_paths",
            added=list(rep_paths - orig_paths),
        )
        return False

    # ── Check 2: GROUP BY presence unchanged ──────────────────────────────────
    orig_has_gb = bool(_HAS_GROUPBY_RE.search(original))
    rep_has_gb = bool(_HAS_GROUPBY_RE.search(repaired))
    if orig_has_gb != rep_has_gb:
        chat_logger.warning(
            "sql_repair_intent_groupby_changed",
            original_had_groupby=orig_has_gb,
            repaired_has_groupby=rep_has_gb,
        )
        return False

    # ── Check 3: GROUP BY columns not removed ─────────────────────────────────
    # Adding a missing column is the legitimate repair case.
    # REMOVING an existing column is never a syntax fix — it changes granularity.
    if orig_has_gb and rep_has_gb:
        orig_gb_tokens = _extract_groupby_tokens(original)
        rep_gb_tokens = _extract_groupby_tokens(repaired)
        removed = orig_gb_tokens - rep_gb_tokens
        if removed:
            chat_logger.warning(
                "sql_repair_intent_groupby_columns_removed",
                removed=list(removed),
                original_tokens=list(orig_gb_tokens),
                repaired_tokens=list(rep_gb_tokens),
            )
            return False

    # ── Check 4: Aggregate count in SELECT unchanged ───────────────────────────
    # A syntax/type repair should never add or remove SUM/COUNT/AVG/etc.
    # If it does, the LLM rewrote the analytical structure, not just the error.
    orig_agg_count = _count_aggregates_in_select(original)
    rep_agg_count = _count_aggregates_in_select(repaired)
    if orig_agg_count != rep_agg_count:
        chat_logger.warning(
            "sql_repair_intent_aggregate_count_changed",
            original_aggregates=orig_agg_count,
            repaired_aggregates=rep_agg_count,
        )
        return False

    # ── Check 5: SELECT * not introduced ─────────────────────────────────────
    if not _HAS_SELECT_STAR_RE.search(original) and _HAS_SELECT_STAR_RE.search(repaired):
        chat_logger.warning("sql_repair_intent_select_star_introduced")
        return False

    return True


def _tier1_repair(sql: str, exc: Exception, null_semantics: dict[str, str]) -> str | None:
    """
    Apply one deterministic rewrite. Returns repaired SQL or None.

    Tries fixes in priority order; stops at the first that changes the SQL.
    Each fix is guarded by a precondition that confirms the rewrite is safe.
    When a precondition cannot be confirmed, the fix is skipped (Tier 2 handles it).
    """
    err = str(exc).lower()
    err_full = str(exc)

    # ── Fix 1: CAST → TRY_CAST for type conversion errors ─────────────────────
    # DuckDB CAST throws on bad input; TRY_CAST returns NULL instead.
    #
    # Delegates to sql_ast.cast_to_try_cast() which uses sqlglot when available
    # (full AST-safe transform) or falls back to the guarded regex (declines on
    # nested CAST).  In both cases, returns None rather than corrupt the SQL.
    if any(tok in err for tok in _ERR_CONVERSION):
        repaired = _sql_ast.cast_to_try_cast(sql)
        if repaired and repaired != sql:
            return repaired

    # ── Fix 2: col = '' → col IS NULL for catalogued null-semantic columns ─────
    # SAP/ERP schemas store "no linked record" as empty string rather than NULL
    # for reference_key columns (e.g. clearing document = '' means open item).
    # Safe ONLY for columns in null_semantics — meaning is explicitly catalogued.
    # We do NOT apply this speculatively to any column with an empty-string test.
    if null_semantics:
        null_cols: set[str] = set()
        for expr in null_semantics:
            # Key format is "TABLE.COL IS NULL" → extract bare column name
            parts = expr.split(" ")[0].split(".")
            null_cols.add(parts[-1].upper())

        repaired = sql
        changed = False
        for m in list(_EMPTY_EQ_RE.finditer(repaired)):
            col_ref = m.group(1)
            bare = col_ref.split(".")[-1].upper()
            if bare in null_cols:
                repaired = repaired.replace(m.group(0), f"{col_ref} IS NULL", 1)
                changed = True
        if changed:
            return repaired

    # ── Fix 3: Add missing column to existing GROUP BY clause ─────────────────
    # DuckDB reports: column "X" must appear in the GROUP BY clause.
    #
    # Guards (both must pass):
    #   a) The column must appear as a bare (non-aggregated) SELECT dimension.
    #      If it's inside an aggregate body, appending it to GROUP BY changes
    #      the analytical granularity in unpredictable ways.
    #   b) The GROUP BY clause must already exist (auto-creating changes semantics).
    #
    # Delegates to sql_ast.append_groupby_column() for the actual rewrite.
    # With sqlglot: AST-level append (safe for CTEs / subqueries).
    # Without sqlglot: guarded regex append.
    if any(tok in err for tok in _ERR_GROUPBY):
        m_err = _GROUPBY_MISSING_RE.search(err_full)
        if m_err:
            missing = m_err.group(1)
            if _column_is_bare_in_select(sql, missing):
                repaired = _sql_ast.append_groupby_column(sql, missing)
                if repaired and repaired != sql:
                    return repaired

    return None


# ── Tier 2: focused LLM repair ─────────────────────────────────────────────────
# Bounded by RepairPolicy — see server/app/policies/repair_policy.py.
_rp = _get_repair_policy()
_REPAIR_OUTPUT_TOKENS = _rp.tier2_output_tokens
_REPAIR_TEMPERATURE   = _rp.tier2_temperature


def _build_context(sql_ctx) -> str:
    """Compact approved-context string injected into repair prompt."""
    lines: list[str] = []

    if getattr(sql_ctx, "approved_joins", None):
        lines.append("Approved joins (ONLY these column pairs are valid):")
        for j in sql_ctx.approved_joins[:8]:
            lines.append(f"  {j.left_table}.{j.left_col} = {j.right_table}.{j.right_col}")

    bindings: dict[str, list[str]] = {}
    bindings.update(getattr(sql_ctx, "column_bindings", {}))
    bindings.update(getattr(sql_ctx, "date_columns", {}))
    if bindings:
        flat = [col for cols in bindings.values() for col in cols]
        lines.append(f"Approved columns: {', '.join(flat[:30])}")

    null_sem = getattr(sql_ctx, "null_semantics", {})
    if null_sem:
        lines.append("NULL semantics (these = IS NULL, not = ''):")
        for expr in list(null_sem)[:6]:
            lines.append(f"  {expr}")

    return "\n".join(lines)


def _tier2_llm_repair(sql: str, exc: Exception, sql_ctx) -> str | None:
    """
    Focused LLM call — fix execution error within approved constraints.
    Returns corrected SQL or None on failure / no-op.
    """
    context = _build_context(sql_ctx)
    error_text = str(exc)[:400]
    sql_text = sql[:1500]

    prompt = (
        "You are fixing a DuckDB SQL execution error. "
        "Return ONLY the corrected SQL — no markdown, no explanation, no preamble.\n\n"
        f"Error:\n{error_text}\n\n"
        f"SQL:\n{sql_text}\n\n"
        "STRICT repair rules (violating these is worse than returning the original):\n"
        "1. Do NOT add tables or files not already in the SQL.\n"
        "2. Do NOT change JOIN columns outside the approved list below.\n"
        "3. Do NOT change the SELECT columns or aggregation logic.\n"
        "4. Fix ONLY the execution error: syntax, types, aliases, NULL handling.\n"
        "5. Use TRY_CAST instead of CAST for all type conversions.\n"
        "6. Use IS NULL / IS NOT NULL — never = '' or != ''.\n"
        "7. Quote reserved DuckDB keywords with double-quotes (e.g. \"order\", \"date\").\n"
        "8. Correct column or alias names using approved columns below.\n"
    )
    if context:
        prompt += f"\nApproved context:\n{context}\n"
    prompt += "\nCorrected SQL:"

    try:
        client, deployment = get_client()
        t = time.perf_counter()
        response = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=_REPAIR_OUTPUT_TOKENS,
            temperature=_REPAIR_TEMPERATURE,
        )
        repaired = (response.choices[0].message.content or "").strip()

        # Strip markdown fences
        if repaired.startswith("```"):
            lines = repaired.split("\n")
            repaired = "\n".join(ln for ln in lines if not ln.strip().startswith("```")).strip()

        # Sanity guards — must be a SELECT and not suspiciously long
        if not repaired.upper().lstrip().startswith("SELECT"):
            chat_logger.warning("sql_repair_llm_not_select", preview=repaired[:80])
            return None
        if len(repaired) > len(sql) * 3:
            chat_logger.warning("sql_repair_llm_too_long",
                                original=len(sql), repaired=len(repaired))
            return None

        pipeline_logger.info(
            "sql_repair_tier2",
            duration_ms=round((time.perf_counter() - t) * 1000, 2),
            original_len=len(sql),
            repaired_len=len(repaired),
        )

        # Verify the repair preserved analytical intent before accepting.
        # GROUP BY presence, file scope, and SELECT structure must be unchanged.
        if not _validate_repair_intent(sql, repaired):
            chat_logger.warning(
                "sql_repair_tier2_intent_rejected",
                original_preview=sql[:120],
                repaired_preview=repaired[:120],
            )
            return None

        return repaired if repaired != sql else None

    except Exception as repair_exc:
        chat_logger.warning("sql_repair_llm_error", error=str(repair_exc)[:200])
        return None


# ── Public API ─────────────────────────────────────────────────────────────────

def attempt_repair(
    sql: str,
    exc: Exception,
    sql_ctx,          # SQLContext | None — passed from build_sql_tools closure
    attempt_number: int,
) -> str | None:
    """
    Attempt to repair a SQL execution failure.

    Args:
        sql:            The SQL that failed (post validate_and_normalise).
        exc:            Exception raised by the query engine.
        sql_ctx:        SQLContext with approved joins/columns. None = no context.
        attempt_number: 0-indexed (0 = first repair, 1 = second repair).

    Returns:
        Repaired SQL string, or None if no repair is possible.

    Attempt strategy:
        attempt 0 → Tier 1 (deterministic, free). If Tier 1 makes no change,
                    fall through to Tier 2 (LLM, ~512 output tokens).
        attempt 1 → Tier 2 only (Tier 1 already tried on attempt 0).
    """
    null_sem = getattr(sql_ctx, "null_semantics", {}) if sql_ctx else {}

    # Attempt 0: try deterministic first (zero cost)
    if attempt_number == 0:
        t1 = _tier1_repair(sql, exc, null_sem)
        if t1 and t1 != sql:
            pipeline_logger.info(
                "sql_repair_tier1",
                error_preview=str(exc)[:120],
            )
            return t1

    # LLM repair — only when we have context to constrain it
    if sql_ctx is None:
        return None

    return _tier2_llm_repair(sql, exc, sql_ctx)
