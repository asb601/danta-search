"""
System prompt builder — assembles the prompt from catalog data,
parquet paths, and conversation context.
"""
from __future__ import annotations

import os
import re
from datetime import date, timedelta

from app.agent.state import MAX_TOOL_CALLS
from app.core.logger import chat_logger
from app.core.token_counter import count_tokens, get_encoding
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


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except (TypeError, ValueError):
        return default


_PROMPT_MODEL = os.getenv("AGENT_PROMPT_TOKEN_MODEL", "gpt-4o-mini")
_SYSTEM_PROMPT_TOKEN_BUDGET = _env_int("AGENT_SYSTEM_PROMPT_TOKEN_BUDGET", 9000)
_CATALOG_TOKEN_BUDGET = _env_int("AGENT_CATALOG_PROMPT_TOKEN_BUDGET", 2200)
_EXECUTION_CONTEXT_TOKEN_BUDGET = _env_int("AGENT_EXECUTION_CONTEXT_TOKEN_BUDGET", 1800)

# Catalog prompt caps. Full schemas remain available through get_file_schema and
# inspect_column; this initial context only needs enough grounding to pick the
# right logical tables without carrying ingestion/debug metadata.
_MAX_PROMPT_TABLES = _env_int("AGENT_PROMPT_CATALOG_TABLE_LIMIT", 10)
_MAX_CSV_ONLY_TABLES = _env_int("AGENT_PROMPT_CSV_TABLE_LIMIT", 4)
_CATALOG_COLUMN_LIMIT = _env_int("AGENT_PROMPT_TABLE_COLUMN_LIMIT", 8)
_PRIORITY_CATALOG_COLUMN_LIMIT = _env_int("AGENT_PROMPT_PRIORITY_COLUMN_LIMIT", 12)

_OBSERVABILITY_ONLY_HEADERS = (
    "CANDIDATE WORKFLOW DECISIONS",
    "WORKFLOW EXPANSION CANDIDATES",
    "REACHABLE JOIN PATHS",
    "ISOLATED FILES",
)


def _token_count(text: str) -> int:
    if not text:
        return 0
    return count_tokens(text, _PROMPT_MODEL)


def _cap_text_tokens(text: str, max_tokens: int) -> str:
    """Hard-cap text by model tokens while preserving the beginning and end."""
    if not text or _token_count(text) <= max_tokens:
        return text
    marker = "\n\n[Runtime prompt budget omitted lower-priority context.]\n\n"
    try:
        enc = get_encoding(_PROMPT_MODEL)
        token_ids = enc.encode(text)
        marker_tokens = enc.encode(marker)
        keep = max(1, max_tokens - len(marker_tokens))
        head = max(1, int(keep * 0.72))
        tail = max(1, keep - head)
        return enc.decode(token_ids[:head]) + marker + enc.decode(token_ids[-tail:])
    except Exception:
        char_budget = max(100, max_tokens * 4)
        head = int(char_budget * 0.72)
        tail = max(1, char_budget - head - len(marker))
        return text[:head] + marker + text[-tail:]


def _one_line(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _clip(value: str, max_chars: int) -> str:
    value = _one_line(value)
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rsplit(" ", 1)[0] + "..."


def _column_names(entry: dict, max_cols: int) -> list[str]:
    cols = [
        str(c.get("name"))
        for c in (entry.get("columns_info") or [])
        if isinstance(c, dict) and c.get("name")
    ]
    if not cols:
        cols = [str(c) for c in (entry.get("column_names") or []) if isinstance(c, str)]
    return cols[:max_cols]


def _range_hints(entry: dict, limit: int = 2) -> list[str]:
    hints = ("year", "date", "period", "month", "fiscal", "quarter", "fy")
    parts: list[str] = []
    for col_name, stats in (entry.get("column_stats") or {}).items():
        if not isinstance(stats, dict):
            continue
        if stats.get("dtype") != "numeric" or not any(h in col_name.lower() for h in hints):
            continue
        mn, mx = stats.get("min"), stats.get("max")
        if mn is not None and mx is not None:
            parts.append(f"{col_name}:{mn}-{mx}")
        if len(parts) >= limit:
            break
    return parts


def _table_line(
    entry: dict,
    logical_table: str,
    *,
    display_name: str | None = None,
    priority: bool = False,
) -> str:
    details: list[str] = []
    if display_name and display_name != logical_table:
        details.append(f"display={display_name}")
    desc = _clip(_neutralize_description(entry.get("ai_description") or ""), _DESC_MAX_CHARS)
    if desc:
        details.append(f"desc={desc}")
    key_dimensions = (entry.get("key_dimensions") or [])[:_DIM_METRIC_LIMIT]
    if key_dimensions:
        details.append(f"dims={', '.join(map(str, key_dimensions))}")
    key_metrics = (entry.get("key_metrics") or [])[:_DIM_METRIC_LIMIT]
    if key_metrics:
        details.append(f"metrics={', '.join(map(str, key_metrics))}")
    dr_start = entry.get("date_range_start")
    dr_end = entry.get("date_range_end")
    if dr_start or dr_end:
        details.append(f"date={dr_start or '?'}..{dr_end or '?'}")
    col_limit = _PRIORITY_CATALOG_COLUMN_LIMIT if priority else _CATALOG_COLUMN_LIMIT
    cols = _column_names(entry, col_limit)
    if cols:
        details.append(f"cols={', '.join(cols)}")
    ranges = _range_hints(entry, 2 if priority else 1)
    if ranges:
        details.append(f"ranges={', '.join(ranges)}")
    if not details:
        return f"  - {logical_table}"
    return f"  - {logical_table}: " + "; ".join(details)


def _fit_lines_to_budget(header: str, lines: list[str], footer: str, max_tokens: int) -> str:
    kept: list[str] = []
    omitted = 0
    for line in lines:
        candidate_lines = [header, *kept, line]
        if footer:
            candidate_lines.append(footer)
        if _token_count("\n".join(candidate_lines)) > max_tokens:
            omitted += 1
            continue
        kept.append(line)
    if omitted:
        kept.append(f"  ... {omitted} lower-priority table(s) omitted; call search_catalog if needed.")
    return "\n".join([header, *kept, footer]).strip()


def _sanitize_execution_context(text: str) -> str:
    """Remove observability-only sections and enforce the execution-context budget."""
    if not text:
        return ""
    lines: list[str] = []
    skip = False
    removed_sections: set[str] = set()
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        if any(upper.startswith(header) for header in _OBSERVABILITY_ONLY_HEADERS):
            skip = True
            removed_sections.add(upper.split("(", 1)[0].strip().rstrip(":"))
            continue
        if skip:
            if not stripped:
                skip = False
                continue
            if stripped == "---" or (stripped.endswith(":") and upper == stripped):
                skip = False
            else:
                continue
        if not skip:
            lines.append(raw_line)
    cleaned = "\n".join(lines).strip()
    if removed_sections:
        chat_logger.info(
            "prompt_context_observability_sections_omitted",
            sections=sorted(removed_sections),
            original_tokens=_token_count(text),
            cleaned_tokens=_token_count(cleaned),
        )
    return _cap_text_tokens(cleaned, _EXECUTION_CONTEXT_TOKEN_BUDGET)


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

Today's date: {today_iso} ({today_human}).
Resolve every relative time expression in the user's question against THIS date,
not against your training cutoff. Examples (assuming today is {today_iso}):
  - "last month"     → the full previous calendar month ({last_month_start} to {last_month_end})
  - "this month"     → {this_month_start} to {today_iso}
  - "YTD" / "this year" → {year_start} to {today_iso}
  - "last year"      → {last_year_start} to {last_year_end}
  - "last 30 days"   → {last_30_start} to {today_iso}
Never invent a date range from a year you remember from training data.

Dataset scope: current authorized catalog.
{shortlist_header}
{parquet_note}
{sample_note}

--- TOOLS ---
1. run_sql             \u2014 Execute SQL only against logical tables that runtime has promoted.
                        Runtime rejects SQL that references columns outside the inspected schema,
                        exhaustive known-value filters that contradict inspected values,
                        unsupported cross-table key predicates, or invented string labels.
2. get_file_schema     \u2014 Returns column names, types, and sample values for a logical table;
                        successful schema inspection promotes that table for SQL when promotion is required.
3. inspect_column      \u2014 Returns dtype, sample values, and a one-line suggested WHERE predicate
                        for a single column. Use this BEFORE writing any filter when you are
                        unsure how the column is stored (year as int vs float, dates as ISO vs
                        delimited month-name string, identifier vs numeric, etc.). Cheap; preferred
                        over guessing or running probe SELECTs.
4. search_catalog      \u2014 Searches authorized discovery metadata, not row values. Use whenever the
                        shortlist above doesn't obviously contain the file you need.
5. inspect_data_format \u2014 Preview raw rows from a discovery candidate; successful inspection can
                        promote that table for SQL when promotion is required.
6. summarise_dataframe \u2014 Compute stats on the last SQL result.
{relations_tool_note}

--- HOW TO WORK ---
Five principles. Apply them to every situation.

1. VERIFY BEFORE YOU ACT
    Before writing any SQL, call get_file_schema on the target logical table (and
    inspect_column for any column whose storage format is unclear — dates,
    codes, years, identifiers). Use only column names and values you actually
     see in those outputs. If the needed business concept is not represented by
     any inspected column, search for a better logical table or state the gap.
     Never assume, guess, or carry over schema knowledge from a previous query.
    Never create SELECT string literals such as 'Open' AS status or translate
    stored codes into business labels unless a field definition or inspected
    data explicitly supports that mapping.
    Do not rename a generic count or amount into a business state. A valid join
    key proves row linkage only; it does not prove approval status, delivery
    status, matching status, or any other requested concept unless an inspected
    column/filter/literal directly represents that concept.
    If the user asks for a year/date but the inspected tables have no usable
    date, year, fiscal, or period column, do not claim the answer is filtered to
    that year. State that the temporal filter is unavailable in the checked schema.
     For multi-part workflow questions, runtime may block
    run_sql until every referenced logical table has been inspected/promoted.

{join_policy_note}

3. EVIDENCE OVER ASSUMPTION
   If a query returns 0 rows, a JOIN fails, or a column is missing: investigate
   the data first (inspect_column, MIN/MAX probe, search_catalog for another
    logical table). "No data found" is the answer of last resort, not the first guess.

4. CHANGE STRATEGY ON FAILURE
   If an approach fails, try something fundamentally different — different file,
   different column, different filter logic. Never retry the same thing with
   only superficial changes (whitespace, quoting, capitalisation).

5. search_catalog searches discovery metadata (logical table names, descriptions, column names).
    It does NOT search row values. To find a row value, filter inside a logical table.

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


_DESC_MAX_CHARS = 90  # max characters shown per file description in the prompt
_DIM_METRIC_LIMIT = 2  # max key_dimensions / key_metrics shown per file


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
        ordered_blobs: list[str] = []
        for entry in catalog:
            blob = entry.get("blob_path")
            if blob and blob in parquet_paths_all and blob not in ordered_blobs:
                ordered_blobs.append(blob)
        for blob in parquet_paths_all:
            if blob not in ordered_blobs:
                ordered_blobs.append(blob)

        lines: list[str] = []
        for blob in ordered_blobs[:_MAX_PROMPT_TABLES]:
            entry = catalog_by_blob.get(blob) or {"blob_path": blob}
            identity = file_identities.identity_for_blob(blob) if file_identities else None
            logical_table = identity.sql_name if identity else logical_name_from_path(blob)
            display_name = identity.logical_name if identity else None
            lines.append(
                _table_line(
                    entry,
                    logical_table,
                    display_name=display_name,
                    priority=(top_blob_paths is None or blob in top_blob_paths),
                )
            )
        if len(ordered_blobs) > _MAX_PROMPT_TABLES:
            lines.append(
                f"  ... {len(ordered_blobs) - _MAX_PROMPT_TABLES} additional shortlisted logical table(s) omitted; use search_catalog."
            )

        header = "Required logical tables in the current shortlist (compact execution view):"
        footer = "Use these logical table names in SQL. Runtime resolves parquet/CSV storage internally."
        note = _fit_lines_to_budget(header, lines, footer, _CATALOG_TOKEN_BUDGET)

        # Also list a small number of CSV-only files (no parquet conversion).
        csv_only = [e for e in catalog if e.get("blob_path") and e["blob_path"] not in parquet_paths_all]
        if csv_only:
            csv_lines: list[str] = []
            for entry in csv_only[:_MAX_CSV_ONLY_TABLES]:
                bp = entry["blob_path"]
                identity = file_identities.identity_for_blob(bp) if file_identities else None
                logical_table = identity.sql_name if identity else logical_name_from_path(bp)
                csv_lines.append(
                    _table_line(
                        entry,
                        logical_table,
                        display_name=(identity.logical_name if identity else None),
                        priority=False,
                    )
                )
            if len(csv_only) > _MAX_CSV_ONLY_TABLES:
                csv_lines.append(f"  ... {len(csv_only) - _MAX_CSV_ONLY_TABLES} CSV-only table(s) omitted; use search_catalog.")
            csv_note = _fit_lines_to_budget(
                "CSV-only logical tables (compact view):",
                csv_lines,
                "These may execute more slowly; use only if relevant.",
                max(300, _CATALOG_TOKEN_BUDGET // 4),
            )
            note = "\n\n".join([note, csv_note])
        return note

    if parquet_blob_path:
        return (
            "Logical table access is available for the selected data. "
            "Use table names from search_catalog or get_file_schema; runtime resolves storage internally."
        )

    return ""


_CONV_CONTEXT_MAX_CHARS = 1200  # cap conversation history to bound token growth


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
    relations_available: bool = True,
) -> str:
    """Assemble the full system prompt for the agent."""
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
    prompt_visible_count = min(shortlist_count, _MAX_PROMPT_TABLES)
    if full_count > shortlist_count:
        shortlist_header = (
            f"Retrieval shortlisted {shortlist_count} of {full_count} ingested files. "
            f"The prompt shows a compact top {prompt_visible_count}; call "
            f"search_catalog to reach omitted shortlist files or the other "
            f"{full_count - shortlist_count} catalog files."
        )
    elif prompt_visible_count < shortlist_count:
        shortlist_header = (
            f"All {full_count} ingested files are authorized. The prompt shows a compact top "
            f"{prompt_visible_count}; call search_catalog to reach omitted files."
        )
    else:
        shortlist_header = f"All {full_count} ingested files are shown below."

    today = date.today()
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

    if relations_available:
        relations_tool_note = """7. extract_relations   — Returns scoped join relationships and, when requested,
                        minimal visible multi-hop paths between selected files.
                        Call only after you have identified the smallest set of
                        tables needed for a multi-file SQL answer. Pass only those
                        logical table names. Start with direct joins; request multi-hop
                        paths only when selected files are not directly connected."""
        join_policy_note = """2. BEFORE ANY MULTI-FILE JOIN, call extract_relations first.
    Use it only for questions that truly need more than one file. Pass the
    smallest selected file set; do not request the global relationship graph.
    Use the returned join_on.file_a_col, join_on.file_b_col, relationship_type,
    path ordering, and join_type from approved relationships directly. Candidate
    or technical_candidate relationships are evidence only: validate them with
    schema/value inspection before joining. If direct relationships are missing
    for selected files, request a bounded multi-hop path. Runtime blocks joins
    between different column names unless an approved relationship supports them.
    If no scoped path is returned, use only strong same-name keys or state that
    the join is not supported by available evidence."""
    else:
        relations_tool_note = """7. extract_relations   — Unavailable for this request because the relationship graph is not trusted."""
        join_policy_note = """2. MULTI-FILE SQL WITH UNTRUSTED RELATIONSHIPS
    Do not call extract_relations and do not use graph relationships for join planning.
    Join tables only when inspected schemas expose a strong same-name key that runtime accepts.
    If the needed files cannot be joined with inspected schema evidence, run separate SQL
    queries per logical table instead of forcing a joined answer. The UI can render each
    successful query as its own result table below the answer."""

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        container_name=container_name,
        max_calls=MAX_TOOL_CALLS,
        parquet_note=parquet_note,
        sample_note=sample_note,
        shortlist_header=shortlist_header,
        shortlist_count=shortlist_count,
        total_file_count=full_count,
        file_override_note=file_override_note,
        relations_tool_note=relations_tool_note,
        join_policy_note=join_policy_note,
        today_iso=today.isoformat(),
        today_human=today.strftime("%A, %d %B %Y"),
        this_month_start=first_of_this_month.isoformat(),
        last_month_start=last_month_start.isoformat(),
        last_month_end=last_month_end.isoformat(),
        year_start=date(today.year, 1, 1).isoformat(),
        last_year_start=last_year_start.isoformat(),
        last_year_end=last_year_end.isoformat(),
        last_30_start=last_30_start.isoformat(),
    )

    # Inject validated SQL context right before the HOW TO WORK behavioural rules
    # so the LLM reads its constraints alongside its work instructions.
    # Workflow topology note (reachable joins + orphaned files) is injected
    # immediately after the SQL context block so the planner sees both together.
    _raw_context_block = ""
    if sql_context_note:
        _raw_context_block = sql_context_note
    if workflow_topology_note:
        _raw_context_block = "\n\n".join(filter(None, [_raw_context_block, workflow_topology_note]))
    _context_block = _sanitize_execution_context(_raw_context_block)
    if _context_block:
        _marker = "--- HOW TO WORK ---"
        if _marker in system_prompt:
            system_prompt = system_prompt.replace(
                _marker, _context_block + "\n\n" + _marker, 1
            )
        else:
            system_prompt += "\n\n" + _context_block

    conversation_tokens = 0
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
            conversation_tokens = _token_count(_ctx)
        system_prompt += (
            "\n\n--- CONVERSATION HISTORY ---\n"
            "The user is continuing a conversation. Use this context to understand "
            "follow-up questions, pronouns ('it', 'that', 'those'), and references "
            "to previous queries or results.\n\n"
            f"{_ctx}\n"
            "---\n"
        )

    final_prompt_was_capped = False
    prompt_tokens_before_hard_cap = _token_count(system_prompt)
    if prompt_tokens_before_hard_cap > _SYSTEM_PROMPT_TOKEN_BUDGET:
        system_prompt = _cap_text_tokens(system_prompt, _SYSTEM_PROMPT_TOKEN_BUDGET)
        final_prompt_was_capped = True

    final_tokens = _token_count(system_prompt)
    chat_logger.info(
        "system_prompt_size",
        chars=len(system_prompt),
        tokens=final_tokens,
        token_budget=_SYSTEM_PROMPT_TOKEN_BUDGET,
        prompt_tokens_before_hard_cap=prompt_tokens_before_hard_cap,
        final_prompt_was_capped=final_prompt_was_capped,
        catalog_tokens=_token_count(parquet_note),
        execution_context_tokens=_token_count(_context_block),
        raw_execution_context_tokens=_token_count(_raw_context_block),
        conversation_tokens=conversation_tokens,
        parquet_file_count=len(parquet_paths_all),
        shown_catalog_file_count=min(len(parquet_paths_all), _MAX_PROMPT_TABLES),
        has_conversation_context=bool(conversation_context),
    )

    return system_prompt


# ── Entity extraction prompt ──────────────────────────────────────────────────

def build_entity_extraction_prompt(query: str) -> str:
    """
    Build the user message for GPT-4o-mini entity extraction.

    No system message — the instruction is embedded directly in the user turn
    to keep the call minimal (matches the llm_tasks single-message convention).
    Output contract: strict JSON {"entities": ["snake_case_noun", ...]}.
    """
    return (
        "Extract business entity nouns from the following query.\n"
        'Return ONLY valid JSON: {"entities": ["entity_1", "entity_2"]}.\n'
        "Normalize to snake_case. No prose, no SQL, no schema names.\n\n"
        f"Query: {query}"
    )
