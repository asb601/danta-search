"""
System prompt builder — assembles the prompt from catalog data,
parquet paths, and conversation context.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from app.agent.state import MAX_TOOL_CALLS
from app.core.config import get_settings
from app.core.logger import chat_logger


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


SYSTEM_PROMPT_TEMPLATE = """{file_override_note}You are a data analyst with DuckDB SQL access to files in Azure Blob Storage.

Today's date: {today_iso} ({today_human}).
Resolve every relative time expression in the user's question against THIS date,
not against your training cutoff. Examples (assuming today is {today_iso}):
  - "last month"     → the full previous calendar month ({last_month_start} to {last_month_end})
  - "this month"     → {this_month_start} to {today_iso}
  - "YTD" / "this year" → {year_start} to {today_iso}
  - "last year"      → {last_year_start} to {last_year_end}
  - "last 30 days"   → {last_30_start} to {today_iso}
Never invent a date range from a year you remember from training data.

Container: {container_name}
{shortlist_header}
{parquet_note}
{sample_note}

--- TOOLS ---
1. run_sql             \u2014 Execute DuckDB SQL.
2. get_file_schema     \u2014 Returns column names, types, and sample values for a file.
3. inspect_column      \u2014 Returns dtype, sample values, and a one-line suggested WHERE predicate
                        for a single column. Use this BEFORE writing any filter when you are
                        unsure how the column is stored (year as int vs float, dates as ISO vs
                        delimited month-name string, identifier vs numeric, etc.). Cheap; preferred
                        over guessing or running probe SELECTs.
4. search_catalog      \u2014 Searches the FULL catalog ({total_file_count} files). Use whenever the
                        shortlist above doesn't obviously contain the file you need.
5. inspect_data_format \u2014 Preview raw rows from a specific file.
6. summarise_dataframe \u2014 Compute stats on the last SQL result.
7. extract_relations   — Returns scoped join relationships and, when requested,
                        minimal visible multi-hop paths between selected files.
                        Call only after you have identified the smallest set of
                        files needed for a multi-file SQL answer. Pass only those
                        blob paths. Start with direct joins; request multi-hop
                        paths only when selected files are not directly connected.

--- HOW TO WORK ---
Five principles. Apply them to every situation.

1. VERIFY BEFORE YOU ACT
   Before writing any SQL, call get_file_schema on the target file (and
   inspect_column for any column whose storage format is unclear — dates,
   codes, years, identifiers). Use only column names and values you actually
   see in those outputs. Never assume, guess, or carry over schema knowledge
   from a previous query.

2. BEFORE ANY MULTI-FILE JOIN, call extract_relations first.
    Use it only for questions that truly need more than one file. Pass the
    smallest selected file set; do not request the global relationship graph.
    Use the returned join_on.file_a_col, join_on.file_b_col, relationship_type,
    path ordering, and join_type from approved relationships directly. Candidate
    or technical_candidate relationships are evidence only: validate them with
    schema/value inspection before joining. If direct relationships are missing
    for selected files, request a bounded multi-hop path. If no scoped path is
    returned, fall back to inspecting columns manually and note the join is
    unverified.

3. EVIDENCE OVER ASSUMPTION
   If a query returns 0 rows, a JOIN fails, or a column is missing: investigate
   the data first (inspect_column, MIN/MAX probe, search_catalog for another
   file). "No data found" is the answer of last resort, not the first guess.

4. CHANGE STRATEGY ON FAILURE
   If an approach fails, try something fundamentally different — different file,
   different column, different filter logic. Never retry the same thing with
   only superficial changes (whitespace, quoting, capitalisation).

5. search_catalog searches metadata (filenames, descriptions, column names).
   It does NOT search row values. To find a row value, filter inside a file.

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
4. **Source** — one short line stating which file(s) the data came from
   and the filter applied.

Do NOT include tabular data in the text — no markdown pipe tables, no CSV rows.
The UI renders the SQL results as an interactive table directly below this
response. Only state numeric totals that are explicitly in the result rows.

If you cannot answer, say so in one sentence and state which files you checked.
Do not ask the user \"would you like me to search\u2026\" \u2014 just go search.

Max {max_calls} tool calls total.
"""


_DESC_MAX_CHARS = 200  # max characters shown per file description in the prompt
_DIM_METRIC_LIMIT = 4  # max key_dimensions / key_metrics shown per file


def build_parquet_note(
    catalog: list[dict],
    parquet_paths_all: dict[str, str],
    parquet_blob_path: str | None,
    container_name: str,
    *,
    top_blob_paths: set[str] | None = None,
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
        for blob, pq in parquet_paths_all.items():
            line = f"  read_parquet('az://{container_name}/{pq}')"
            entry = catalog_by_blob.get(blob)

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

            # Surface date range so LLM knows what period the file covers
            dr_start = entry.get("date_range_start") if entry else None
            dr_end = entry.get("date_range_end") if entry else None
            if dr_start or dr_end:
                line += f"\n    Date range: {dr_start or '?'} \u2192 {dr_end or '?'}"

            # Surface column stats only for top-retrieved files to keep prompt
            # token load bounded. Lower-ranked files get date range only.
            _is_priority = top_blob_paths is None or blob in top_blob_paths
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
            "Initial shortlist of likely parquet files:\n"
            + "\n".join(lines)
            + "\nParquet covers the FULL dataset. Use it for any ordering, filtering, counting, or row retrieval."
        )

        # Also list CSV-only files (no parquet conversion)
        csv_only = [e for e in catalog if e.get("blob_path") and e["blob_path"] not in parquet_paths_all]
        if csv_only:
            sample_rows = max(1, int(get_settings().INGEST_DUCKDB_SAMPLE_ROWS))
            csv_lines = []
            for entry in csv_only:
                bp = entry["blob_path"]
                csv_line = f"  read_csv_auto('az://{container_name}/{bp}', sample_size={sample_rows}, null_padding=true, ignore_errors=true)"
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
                "\n\nCSV-only files (no parquet — may be slower for large files):\n"
                + "\n".join(csv_lines)
            )
        return note

    if parquet_blob_path:
        return (
            f"Parquet path (use directly in run_sql — no search_catalog needed):\n"
            f"  read_parquet('az://{container_name}/{parquet_blob_path}')"
            "\nParquet covers the FULL dataset. Use it for any ordering, filtering, counting, or row retrieval."
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
) -> str:
    """Assemble the full system prompt for the agent."""
    parquet_note = build_parquet_note(
        catalog, parquet_paths_all, parquet_blob_path, container_name,
        top_blob_paths=top_blob_paths,
    )

    sample_note = ""
    if sample_rows_by_blob:
        sample_note = (
            f"\nData format preview: ingest-time example rows are available for {len(sample_rows_by_blob)} files via"
            " inspect_data_format(blob_path, n=5) — use this only after you know which file you want to inspect."
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
            f"semantic matching. Call get_file_schema on {names} first, then run SQL on it.\n\n"
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
    )

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
    Output contract: strict JSON {"entities": ["snake_case_noun", ...]}.
    """
    return (
        "Extract business entity nouns from the following query.\n"
        'Return ONLY valid JSON: {"entities": ["entity_1", "entity_2"]}.\n'
        "Normalize to snake_case. No prose, no SQL, no schema names.\n\n"
        f"Query: {query}"
    )
