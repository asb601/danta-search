"""
System prompt builder — assembles the prompt from catalog data,
parquet paths, and conversation context.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from app.agent.state import MAX_TOOL_CALLS
from app.core.logger import chat_logger
from app.services.file_identity import FileIdentityMap, logical_name_from_path


# Auto-generated descriptions often start with absolutist phrases like
# "This file is the PRIMARY source for..." or "Unlike similar files, this
# file...". Those phrases over-anchor the LLM and stop it from considering
# alternative files in the catalog. We strip them at render time so the
# stored description is unchanged but the prompt sees neutral text.
_ANCHOR_PATTERNS = [
    re.compile(r"\bThis file is the PRIMARY source\b", re.IGNORECASE),
    re.compile(r"\bThis file is THE PRIMARY source\b", re.IGNORECASE),
    re.compile(r"\bPRIMARY source\b"),
    re.compile(r"\bUnlike (?:other|similar) files,?\s*", re.IGNORECASE),
    re.compile(r"\bnot (?:typically )?found in other (?:similar )?files\b", re.IGNORECASE),
]


def _neutralize_description(desc: str) -> str:
    """Remove over-anchoring phrases from auto-generated descriptions."""
    if not desc:
        return ""
    out = desc
    for pat in _ANCHOR_PATTERNS:
        out = pat.sub("", out)
    # Collapse double spaces and stray leading punctuation introduced by removals
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"^[,;:.\s]+", "", out)
    return out


SYSTEM_PROMPT_TEMPLATE = """{file_override_note}You are a data analyst with read-only SQL access to logical tables.

Runtime owns storage: never write blob paths, parquet filenames, physical storage
URIs, or storage scan functions. Use only the logical table names
shown below in FROM/JOIN clauses; the runtime resolves them to authorized files.
Each logical table already spans all of its time periods — query the table name
once; do NOT union per-month tables or assume a period is "missing" from one file.

{sql_dialect_note}

Reference date for relative time: {today_iso} ({today_human}) — the most recent
date this dataset covers. Resolve every relative time expression in the user's
question against THIS date (not the wall clock, not your training cutoff, and NOT
SQL current_date, which may be later than the data). Examples:
  - "last month"        → the full previous calendar month ({last_month_start} to {last_month_end})
  - "this month" / MTD  → {this_month_start} to {today_iso}
  - "YTD" / "this year"  → {year_start} to {today_iso}
  - "last year"         → {last_year_start} to {last_year_end}
  - "last 30 days"      → {last_30_start} to {today_iso}
For relative-time filters use these explicit dates, not current_date.
Never invent a date range from a year you remember from training data.

COUNTING & DISTINCTNESS: "how many / number of / distinct <entity>" means
COUNT(DISTINCT <entity key>). A tool result's total_rows is a ROW count — the
same entity can repeat across rows (e.g. one VENDOR_ID with several names) — so
never report total_rows as the count of distinct entities. To count entities,
COUNT(DISTINCT <id column>); a multi-column SELECT DISTINCT counts distinct row
combinations, not distinct entities.

Dataset scope: current authorized catalog.
{shortlist_header}
{parquet_note}
{sample_note}

--- TOOLS ---
1. run_sql             \u2014 Execute SQL against logical tables.
2. get_file_schema     \u2014 Returns column names, types, and sample values for a logical table.
3. inspect_column      \u2014 Returns dtype, sample values, and a suggested WHERE predicate for a
                        single column. Use before any filter whose storage format is unclear
                        (dates, codes, years, identifiers). Cheap — prefer over guessing.
4. search_catalog      \u2014 Searches the FULL catalog ({total_file_count} files) by metadata
                        (table names, descriptions, column names). It does NOT search row
                        values. Use when the shortlist doesn't clearly contain what you need.
5. inspect_data_format \u2014 Preview raw rows from a specific logical table.
6. summarise_dataframe \u2014 Compute stats on the last SQL result.
7. extract_relations   \u2014 Returns scoped join relationships and bounded multi-hop paths.
                        Pass the smallest set of tables you have selected. Start with
                        direct joins; request multi-hop only when selected files are not
                        directly connected.

--- ANALYST WORKFLOW ---
Work through every data request in four phases.

PHASE 1 \u2014 DECOMPOSE
Identify (a) the primary business subject and (b) each requested facet or section.
Anchor everything on the primary subject. Treat additional facets as evidence
requirements attached to that anchor, not as permission to explore unrelated domains.
For "analyze" or "summarize" requests: plan aggregate SQL from the start, not row
detail, unless the user explicitly asks to list records.

PHASE 2 \u2014 GROUND  (mandatory before writing any SQL)
Inspect schemas before assuming anything.
\u2022 Call get_file_schema on the primary shortlisted file(s) first.
\u2022 Call inspect_column for any column whose format is unclear.
\u2022 For each requested facet, check the schemas you just inspected: if the required
  column already exists there, use it directly. Only call search_catalog for a facet
  when no already-inspected schema contains the needed column.
Schema knowledge from previous queries is stale \u2014 always re-inspect in this request.

PHASE 3 \u2014 CONNECT  (only when the answer genuinely needs more than one file)
Call extract_relations with the smallest possible file set.
Use the returned join columns directly. Candidate/technical_candidate relationships
are unverified \u2014 inspect column values before joining on them. If no direct
relationship exists, request a bounded multi-hop path; if none is returned, join
manually and flag it as unverified in your response.

PHASE 4 \u2014 EXECUTE AND ADAPT
Write SQL using only column names verified in Phase 2, and only filter VALUES you
have actually observed (via inspect_column / get_file_schema). Never invent a
status/category literal (e.g. do not assume a value 'Shipped' exists).
\u2022 0 rows on an aggregation/filter is NOT permission to switch tables or domains.
  First, on the SAME table(s): (a) drop your most specific filter and re-run;
  (b) probe the date column with SELECT MIN(col), MAX(col) to confirm your window
  overlaps the data \u2014 the data may simply not cover the requested period; (c)
  SELECT DISTINCT the status/category column you filtered to confirm the literal
  exists. Only after these may you conclude the value/period is genuinely absent.
\u2022 Switching to a table from a DIFFERENT business domain after a 0-row aggregation
  is almost always wrong \u2014 do not join across unrelated domains to manufacture rows.
\u2022 JOIN fails \u2192 re-examine the join column with inspect_column.
\u2022 Column missing \u2192 search_catalog for an alternative table.
Never retry the same query with only cosmetic changes \u2014 change the approach.

TREND / DIVERGENCE QUESTIONS (one metric worsening while another rises, over time):
\u2022 Normalise both metrics to ONE period grain before comparing. If one table is at
  month/YYYYMM and another coarser, roll the finer up (quarter = 'Q' || CAST(CEIL(MONTH/3) AS INT)).
  Never compare metrics at mismatched grains.
\u2022 If the two metrics live in tables that do not share the comparison dimension, route
  the metric-bearing table through the bridge table named in WORKFLOW TOPOLOGY to acquire
  that dimension BEFORE grouping (e.g. a deliveries table lacking a region dimension joins
  through an orders table that carries both the plant and the region).
\u2022 Compute period-over-period change per dimension with a window:
  metric - LAG(metric) OVER (PARTITION BY <dim> ORDER BY <period>).
\u2022 "Divergence" = periods where the two deltas move in the worsening direction together;
  rank by joint magnitude to find the sharpest period. Do NOT answer with two independent
  aggregates \u2014 the question asks whether they track each other.

TWO CORE PRINCIPLES (apply across all phases)
A. Evidence is not transferable. Delivery status \u2260 approval status \u2260 payment status.
   Each facet requires its own data evidence even if the concepts sound related.
B. "No data found" is a last resort, not a first answer. Investigate before giving up.

--- QUESTION TYPE ---
Conceptual ("how does X work", "explain Y"): answer from knowledge + file
descriptions. No SQL needed unless you need a column list.
Data ("show me", "how many", "top N", filters): run SQL using the steps above.

--- OUTPUT STYLE (MANDATORY) ---
Do NOT narrate your reasoning, plans, or next steps (no \"Let me start by\u2026\",
\"Plan: 1. \u2026\", \"I'll now query\u2026\"). Reasoning happens silently via tool calls.

When you finish, write a complete analyst response:

1. **Direct answer** \u2014 one sentence that directly answers the question
    (e.g. \"The top 5 records by outstanding balance total $4.2M across
    312 open items.\").
2. **Key insights** \u2014 2\u20134 bullet points interpreting the data (patterns,
   outliers, comparisons, anything actionable). Write as a business analyst.
3. **Table note** — if SQL returned rows, end with the line:
   "↓ See the results table below for the full data."
4. **Source** — one short line stating which logical table(s) the data came from
   and the filter applied.

Do NOT include tabular data in the text — no markdown pipe tables, no CSV rows.
The UI renders the SQL results as an interactive table directly below this
response. Only state numeric totals that are explicitly in the result rows.

If you cannot answer, say so in one sentence and state which logical tables you checked.
Do not ask the user \"would you like me to search\u2026\" \u2014 just go search.

Max {max_calls} tool calls total.
"""


def _sql_dialect_note() -> str:
    """SQL-dialect guidance matching the ACTIVE execution engine (QUERY_ENGINE).

    The agent must write SQL for whatever engine tools/sql.py::_execute runs, so
    this is derived from config rather than hardcoded — keeping the prompt and the
    executor in sync. The RESOLVE fast path emits its own engine-agnostic SQL and
    does not depend on this note.
    """
    from app.core.config import get_settings  # noqa: PLC0415

    if get_settings().QUERY_ENGINE == "datafusion":
        return (
            "SQL dialect: the executor is Apache DataFusion (Arrow-native, ANSI SQL). "
            "Write standard ANSI SQL. Use current_date for today; date_part(...)/"
            "extract(...)/date_trunc(...) for date handling; CAST(... AS ...) for "
            "conversions. When a value already exists as a column (e.g. an aging/"
            "days/DSO column), use it directly rather than recomputing from current_date."
        )
    return (
        "SQL dialect: the executor is DuckDB. Write DuckDB-valid SQL. Use "
        "date_diff('day', a, b) (not DATEDIFF), string_agg(x, ',') (not GROUP_CONCAT), "
        "and current_date. When a value already exists as a column (e.g. an aging/"
        "days/DSO column), use that column directly rather than recomputing it from "
        "current_date."
    )


_DESC_MAX_CHARS = 200  # max characters shown per file description in the prompt
_DIM_METRIC_LIMIT = 4  # max key_dimensions / key_metrics shown per file
_PROMPT_COLUMN_LIMIT = 40  # exact columns shown for priority files


# ── SME join-enforcement prompt swap (flag-gated, byte-identical when off) ─────
# When relationship-graph join enforcement is active, the execution layer will
# REJECT any JOIN whose table pair is not an approved relationship. The default
# Phase-3 guidance below licenses the model to "join manually and flag it as
# unverified" — which directly contradicts that enforcement (the manual join is
# rejected, not flagged). When the flag is on we replace ONLY that sentence with
# the approved-path-or-independent instruction, leaving the rest of the prompt
# verbatim. When the flag is off, no substitution runs and the prompt is the
# unchanged template (byte-identical).
_JOIN_LICENSE_ORIGINAL = (
    "If no direct\n"
    "relationship exists, request a bounded multi-hop path; if none is returned, join\n"
    "manually and flag it as unverified in your response."
)
_JOIN_LICENSE_ENFORCED = (
    "If no direct\n"
    "relationship exists, request a bounded multi-hop path; if none is returned, do NOT\n"
    "join the tables — analyze each table independently and state that no validated\n"
    "relationship exists between them. A manual/invented join will be rejected at execution."
)


def _column_names_for_prompt(entry: dict | None) -> list[str]:
    if not entry:
        return []
    names: list[str] = []
    for col in entry.get("columns_info") or []:
        if isinstance(col, dict) and col.get("name"):
            names.append(str(col["name"]))
        elif isinstance(col, str):
            names.append(col)
    if not names:
        names = [str(c) for c in (entry.get("column_names") or []) if isinstance(c, str)]
    return names[:_PROMPT_COLUMN_LIMIT]


def build_parquet_note(
    catalog: list[dict],
    parquet_paths_all: dict[str, str],
    parquet_blob_path: str | None,
    container_name: str,
    *,
    top_blob_paths: set[str] | None = None,
    file_identities: FileIdentityMap | None = None,
) -> str:
    """Build the file-listing section of the system prompt.

    top_blob_paths: blobs that receive full column_stats context (top-N by
    retrieval score). All other files get a compact description to reduce
    total token load without removing any file from the shortlist.
    """
    catalog_by_blob: dict[str, dict] = {}
    for entry in catalog:
        bp = entry.get("blob_path")
        if bp:
            catalog_by_blob[bp] = entry

    if parquet_paths_all:
        lines = []
        seen_logical: set[str] = set()
        for blob, pq in parquet_paths_all.items():
            entry = catalog_by_blob.get(blob)
            identity = file_identities.identity_for_blob(blob) if file_identities else None
            logical_table = identity.sql_name if identity else logical_name_from_path(blob)
            # Consolidation: a logical table spans many partition blobs. Emit ONE
            # line per logical table (not per month) so the model sees a single
            # name with the TRUE coverage span — never 36 lines with conflicting
            # single-month ranges (which previously triggered false "missing" claims).
            if logical_table in seen_logical:
                continue
            seen_logical.add(logical_table)
            line = f"  {logical_table}"
            if identity and identity.sql_name != identity.logical_name:
                line += f"  (display: {identity.logical_name})"
            if identity and identity.partition_count > 1:
                line += f"\n    Partitions: {identity.partition_count} (one logical table; query the name once)"

            desc = _neutralize_description(entry.get("ai_description") if entry else "")
            if desc:
                # Truncate description to keep per-file token cost bounded.
                if len(desc) > _DESC_MAX_CHARS:
                    desc = desc[:_DESC_MAX_CHARS].rsplit(" ", 1)[0] + "…"
                line += f"\n    Description: {desc}"
            key_dimensions = (entry.get("key_dimensions") or []) if entry else []
            if key_dimensions:
                line += f"\n    Key dimensions: {', '.join(key_dimensions[:_DIM_METRIC_LIMIT])}"
            key_metrics = (entry.get("key_metrics") or []) if entry else []
            if key_metrics:
                line += f"\n    Key metrics: {', '.join(key_metrics[:_DIM_METRIC_LIMIT])}"

            # Surface the AGGREGATE coverage window across all partitions (from
            # the consolidated identity) so the model knows the full span, not one
            # month. Fall back to the single entry's range when no identity.
            if identity and (identity.coverage_start or identity.coverage_end):
                dr_start, dr_end = identity.coverage_start, identity.coverage_end
            else:
                dr_start = entry.get("date_range_start") if entry else None
                dr_end = entry.get("date_range_end") if entry else None
            if dr_start or dr_end:
                line += f"\n    Date range (full coverage): {dr_start or '?'} \u2192 {dr_end or '?'}"

            # Surface column stats only for top-retrieved files to keep prompt
            # token load bounded. Lower-ranked files get date range only.
            _is_priority = top_blob_paths is None or blob in top_blob_paths
            if _is_priority:
                columns = _column_names_for_prompt(entry)
                if columns:
                    line += f"\n    Available columns: {', '.join(columns)}"

            if _is_priority:
                _DATE_HINTS = ("year", "date", "period", "month", "fiscal", "quarter", "fy")
                col_stats = (entry.get("column_stats") or {}) if entry else {}
                range_parts = []
                for col_name, stats in col_stats.items():
                    if stats.get("dtype") == "numeric" and any(
                        h in col_name.lower() for h in _DATE_HINTS
                    ):
                        mn, mx = stats.get("min"), stats.get("max")
                        if mn is not None and mx is not None:
                            range_parts.append(f"{col_name}: {mn}\u2013{mx}")
                if range_parts:
                    line += f"\n    Column ranges: {', '.join(range_parts[:4])}"

            lines.append(line)

        note = (
            "Initial shortlist of likely logical tables:\n"
            + "\n".join(lines)
            + "\nUse these names directly in SQL FROM/JOIN clauses. Parquet/CSV storage is resolved internally."
        )

        # Also list CSV-only files (no parquet conversion)
        csv_only = [e for e in catalog if e.get("blob_path") and e["blob_path"] not in parquet_paths_all]
        if csv_only:
            csv_lines = []
            for entry in csv_only:
                bp = entry["blob_path"]
                identity = file_identities.identity_for_blob(bp) if file_identities else None
                logical_table = identity.sql_name if identity else logical_name_from_path(bp)
                if logical_table in seen_logical:
                    continue
                seen_logical.add(logical_table)
                csv_line = f"  {logical_table}"
                desc = _neutralize_description(entry.get("ai_description") or "")
                if desc:
                    csv_line += f"\n    Description: {desc}"
                # Note: leave key_dimensions / key_metrics intact below.
                key_dimensions = entry.get("key_dimensions") or []
                if key_dimensions:
                    csv_line += f"\n    Key dimensions: {', '.join(key_dimensions[:6])}"
                key_metrics = entry.get("key_metrics") or []
                if key_metrics:
                    csv_line += f"\n    Key metrics: {', '.join(key_metrics[:6])}"
                csv_lines.append(csv_line)
            note += (
                "\n\nCSV-only logical tables (runtime may execute them more slowly):\n"
                + "\n".join(csv_lines)
            )
        return note

    if parquet_blob_path:
        return (
            "Logical table access is available for the selected data. "
            "Use table names from search_catalog or get_file_schema; runtime resolves storage internally."
        )

    return ""


_CONV_CONTEXT_MAX_CHARS = 2000  # cap conversation history to bound token growth


def build_system_prompt(
    catalog: list[dict],
    parquet_paths_all: dict[str, str],
    parquet_blob_path: str | None,
    container_name: str,
    sample_rows_by_blob: dict[str, list],
    conversation_context: str = "",
    total_file_count: int | None = None,
    mentioned_files: list[str] | None = None,
    sql_context_note: str = "",
    *,
    top_blob_paths: set[str] | None = None,
    workflow_topology_note: str = "",
    file_identities: FileIdentityMap | None = None,
    as_of_date: date | None = None,
) -> str:
    """Assemble the full system prompt for the agent.

    as_of_date — the data-driven reference 'now' for relative-time resolution
    (the dataset's latest coverage date). Falls back to the wall clock only when
    the catalog carries no date coverage. See resolve_as_of_date().
    """
    parquet_note = build_parquet_note(
        catalog, parquet_paths_all, parquet_blob_path, container_name,
        top_blob_paths=top_blob_paths,
        file_identities=file_identities,
    )

    sample_note = ""
    if sample_rows_by_blob:
        sample_note = (
            f"\nData format preview: ingest-time example rows are available for {len(sample_rows_by_blob)} files via"
            " inspect_data_format(logical_table, n=5) — use this only after you know which table you want to inspect."
        )

    shortlist_count = len(catalog)
    full_count = total_file_count if total_file_count is not None else shortlist_count
    if full_count > shortlist_count:
        shortlist_header = (
            f"Showing the top {shortlist_count} of {full_count} ingested files "
            f"(retrieval-ranked for this query). The other "
            f"{full_count - shortlist_count} files are NOT shown — call "
            f"search_catalog to reach them."
        )
    else:
        shortlist_header = f"All {full_count} ingested files are shown below."

    today = as_of_date or date.today()
    # Last calendar month bounds
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    # Last calendar year bounds
    last_year_start = date(today.year - 1, 1, 1)
    last_year_end = date(today.year - 1, 12, 31)
    # Last 30 days
    last_30_start = today - timedelta(days=30)

    if mentioned_files:
        names = ", ".join(f"`{f}`" for f in mentioned_files)
        file_override_note = (
            f"USER SPECIFIED FILE: {names}\n"
            f"Query ONLY this file. Do not redirect to a different file based on "
            f"semantic matching. Call get_file_schema on {names} first, then run logical SQL on it.\n\n"
        )
    else:
        file_override_note = ""

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        container_name=container_name,
        max_calls=MAX_TOOL_CALLS,
        parquet_note=parquet_note,
        sample_note=sample_note,
        shortlist_header=shortlist_header,
        shortlist_count=shortlist_count,
        total_file_count=full_count,
        file_override_note=file_override_note,
        today_iso=today.isoformat(),
        today_human=today.strftime("%A, %d %B %Y"),
        this_month_start=first_of_this_month.isoformat(),
        last_month_start=last_month_start.isoformat(),
        last_month_end=last_month_end.isoformat(),
        year_start=date(today.year, 1, 1).isoformat(),
        last_year_start=last_year_start.isoformat(),
        last_year_end=last_year_end.isoformat(),
        last_30_start=last_30_start.isoformat(),
        sql_dialect_note=_sql_dialect_note(),
    )

    # SME join enforcement: when joins are rejected at execution, the prompt must
    # not license a manual/unverified join. Swap that one sentence (flag-gated).
    # Default OFF → no substitution → byte-identical prompt.
    try:
        from app.core.config import get_settings as _gs  # noqa: PLC0415
        _s = _gs()
        if (
            getattr(_s, "SME_MODE_ENABLED", False)
            and getattr(_s, "SME_JOIN_ENFORCE_ENABLED", False)
            and _JOIN_LICENSE_ORIGINAL in system_prompt
        ):
            system_prompt = system_prompt.replace(
                _JOIN_LICENSE_ORIGINAL, _JOIN_LICENSE_ENFORCED, 1
            )
    except Exception as _exc:  # never let prompt assembly fail on a flag read
        chat_logger.warning("sme_join_prompt_swap_error", error=str(_exc)[:200])

    # Inject validated SQL context right before the HOW TO WORK behavioural rules
    # so the LLM reads its constraints alongside its work instructions.
    # Workflow topology note (reachable joins + orphaned files) is injected
    # immediately after the SQL context block so the planner sees both together.
    _context_block = ""
    if sql_context_note:
        _context_block = sql_context_note
    if workflow_topology_note:
        _context_block = "\n\n".join(filter(None, [_context_block, workflow_topology_note]))
    if _context_block:
        _marker = "--- HOW TO WORK ---"
        if _marker in system_prompt:
            system_prompt = system_prompt.replace(
                _marker, _context_block + "\n\n" + _marker, 1
            )
        else:
            system_prompt += "\n\n" + _context_block

    if conversation_context:
        # Truncate conversation history to bound per-request token cost.
        # Long conversations grow linearly; most context is in the last few turns.
        _ctx = conversation_context
        if len(_ctx) > _CONV_CONTEXT_MAX_CHARS:
            _ctx = _ctx[-_CONV_CONTEXT_MAX_CHARS:]
            # Avoid starting mid-word after truncation
            nl = _ctx.find("\n")
            if 0 < nl < 200:
                _ctx = _ctx[nl + 1:]
        system_prompt += (
            "\n\n--- CONVERSATION HISTORY ---\n"
            "The user is continuing a conversation. Use this context to understand "
            "follow-up questions, pronouns ('it', 'that', 'those'), and references "
            "to previous queries or results.\n\n"
            f"{_ctx}\n"
            "---\n"
        )

    chat_logger.info("system_prompt_size",
                     chars=len(system_prompt),
                     words=len(system_prompt.split()),
                     parquet_file_count=len(parquet_paths_all),
                     has_conversation_context=bool(conversation_context))

    return system_prompt


# ── Entity extraction prompt ──────────────────────────────────────────────────

def build_entity_extraction_prompt(query: str) -> str:
    """
    Build the user message for GPT-4o-mini entity extraction.

    No system message — the instruction is embedded directly in the user turn
    to keep the call minimal (matches the llm_tasks single-message convention).
    Output contract: strict JSON {"entities": ["snake_case_concept", ...]}.
    """
    return (
        "Extract the business objects, processes, workflow states, exceptions, and "
        "relationships that a data agent needs to find the right tables for this query.\n"
        'Return ONLY valid JSON: {"entities": ["snake_case_concept"]}.\n'
        "Rules:\n"
        "1. Expand abbreviations (PO → purchase_order, SO → sales_order, etc.).\n"
        "2. Include workflow states, exceptions, matching/reconciliation, holds, and "
        "lifecycle events from colon sections and bullet lists when they need their own data.\n"
        "3. Anchor generic labels to their owner: po_approval_status not approval_status.\n"
        "4. Exclude: time ranges, metrics/values, display fields, recommendations, "
        "next actions, and output instructions.\n"
        "5. Return up to 10 concise singular snake_case entities.\n\n"
        f"Query: {query}"
    )
