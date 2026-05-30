"""High-level LLM tasks used by backend services."""
from __future__ import annotations

import asyncio
import json
import re
import time

from app.core.config import get_settings
from app.core.logger import chat_logger, ingest_logger
from app.core.openai_client import get_client
from app.core.token_counter import count_tokens, elapsed_ms, track_and_log


_ENTITY_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_ENTITY_DETAIL_TOKENS = frozenset({"detail", "details", "record", "records", "row", "rows"})
_ENTITY_NOISE_TOKENS = frozenset({
    "action",
    "actions",
    "approval",
    "approvals",
    "count",
    "current",
    "date",
    "issue",
    "issues",
    "matching",
    "month",
    "next",
    "pending",
    "problem",
    "problems",
    "quarter",
    "recommendation",
    "recommendations",
    "recommended",
    "status",
    "statuses",
    "summary",
    "time",
    "total",
    "value",
    "year",
})
_ENTITY_FACET_SUFFIXES = frozenset({
    "action",
    "actions",
    "approval",
    "approvals",
    "issue",
    "issues",
    "problem",
    "problems",
    "recommendation",
    "recommendations",
    "status",
    "statuses",
})
_SUMMARY_SECTION_RE = re.compile(r"\b(?:summari[sz]e|including|include|covering|with)\b\s*[:\-]", re.I)


def _entity_tokens(text: str) -> list[str]:
    return [t for t in _ENTITY_TOKEN_RE.split(text.lower()) if t]


def _entity_acronym(tokens: list[str]) -> str:
    return "".join(t[0] for t in tokens if t)


def _primary_query_segment(query: str) -> tuple[str, bool]:
    marker = _SUMMARY_SECTION_RE.search(query)
    if marker:
        return query[:marker.start()], True
    if ":" in query:
        return query.split(":", 1)[0], True
    return query, False


def _mentioned_in_primary_segment(tokens: list[str], primary_text: str) -> bool:
    primary_tokens = set(_entity_tokens(primary_text))
    if not primary_tokens:
        return False
    acronym = _entity_acronym(tokens) if len(tokens) > 1 else ""
    return all(t in primary_tokens for t in tokens) or bool(acronym and acronym in primary_tokens)


def _clean_extracted_entities(entities: list, query: str = "") -> list[str]:
    """Keep only durable business objects; drop output facets and time terms."""
    cleaned: list[str] = []
    seen: set[str] = set()
    primary_text, has_summary_sections = _primary_query_segment(query or "")
    primary_subject_tokens = {
        t for t in _entity_tokens(primary_text)
        if t not in _ENTITY_NOISE_TOKENS and t not in _ENTITY_DETAIL_TOKENS
    }
    enforce_primary_segment = has_summary_sections and bool(primary_subject_tokens)
    for raw in entities:
        tokens = _entity_tokens(str(raw))
        while tokens and tokens[-1] in _ENTITY_DETAIL_TOKENS:
            tokens.pop()
        if not tokens:
            continue
        if enforce_primary_segment and not _mentioned_in_primary_segment(tokens, primary_text):
            continue
        if all(t in _ENTITY_NOISE_TOKENS for t in tokens):
            continue
        if len(tokens) > 1 and tokens[-1] in _ENTITY_FACET_SUFFIXES:
            continue
        entity = "_".join(tokens)
        if entity and entity not in seen:
            cleaned.append(entity)
            seen.add(entity)
    return cleaned[:4]


def safe_parse_json(text: str) -> dict:
    """Strip markdown fences and parse JSON. Returns fallback on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return {}


async def generate_file_description(
    columns_info: list,
    sample_rows: list,
    filename: str,
    domain_tag: str | None = None,
    column_glossary: dict[str, str] | None = None,
) -> dict:
    def _run() -> dict:
        settings = get_settings()
        description_sample_rows = max(0, int(settings.INGEST_FILE_DESCRIPTION_SAMPLE_ROWS))
        max_completion_tokens = max(1, int(settings.INGEST_FILE_DESCRIPTION_MAX_COMPLETION_TOKENS))
        raw_retry_delays = settings.INGEST_LLM_RETRY_DELAYS_SECONDS
        retry_delay_items = raw_retry_delays.split(",") if isinstance(raw_retry_delays, str) else raw_retry_delays
        retry_delays = [
            max(0, int(str(raw).strip()))
            for raw in retry_delay_items
            if str(raw).strip()
        ]
        client, deployment = get_client()
        cols_for_prompt = [
            {
                "name": c["name"],
                "type": c["type"],
                "samples": c["sample_values"],
                "unique": c["unique_values"],
            }
            for c in columns_info
        ]

        # Build domain + glossary context injected into the prompt
        context_block = ""
        if domain_tag:
            context_block += f'\nDomain context: This file belongs to the "{domain_tag}" domain.'
        if column_glossary:
            glossary_lines = "\n".join(
                f"  {code} = {meaning}" for code, meaning in column_glossary.items()
            )
            # Build the set of glossary keys (case-insensitive match basis)
            glossary_keys = {k.strip().lower() for k in column_glossary.keys()}
            # Identify columns in this file that are NOT covered by the glossary
            uncovered = [
                c["name"] for c in columns_info
                if c.get("name") and c["name"].strip().lower() not in glossary_keys
            ]
            uncovered_clause = ""
            if uncovered:
                uncovered_preview = ", ".join(uncovered[:description_sample_rows])
                uncovered_clause = (
                    "\n\nColumns NOT in the glossary (use the raw column name AS-IS — "
                    "do NOT invent or guess a business meaning for these; if a column's "
                    f"purpose is unclear, describe it neutrally from its sample values): {uncovered_preview}"
                )
            context_block += (
                "\nColumn name glossary — for any column listed below, ALWAYS write "
                "the business term first and put the raw code in parentheses, e.g. "
                "'Amount in Local Currency (WRBTR)'. Never use the raw code alone when "
                "a glossary entry exists:\n" + glossary_lines + uncovered_clause
            )
        if context_block:
            context_block = "\n" + context_block.strip()

        prompt = f"""You are a data catalog expert analyzing a file named "{filename}".
Your output will be used to match natural language business questions to the correct file.
Be SPECIFIC and DISCRIMINATIVE — your description must distinguish this file from other similar files.{context_block}

Return ONLY this JSON with no preamble no markdown:
{{
    "summary": "2 sentences: (1) what specific business questions this file can support, naming the exact columns that make it useful (e.g. 'AMOUNT_VALUE tracks balance', 'STATUS_CODE filters active records'); (2) what this file contains that similar-sounding files do not have. Avoid claiming this is the only, best, or primary source unless that is explicitly present in metadata.",
    "good_for": ["3-6 exact natural language business question phrases this file can support. Use the actual business domain terms, column names, and metrics specific to this file — do NOT use generic placeholder terms. Make each phrase discriminative: it should clearly distinguish this file from other similar-sounding files in the catalog without claiming exclusivity."],
  "key_metrics": ["numeric columns used for aggregation and SUM/AVG calculations"],
  "key_dimensions": ["categorical, status, and ID columns used for filtering and grouping — include their important values where relevant, e.g. STATUS: OP=open CL=closed"],
  "date_range_start": "YYYY-MM-DD or null",
  "date_range_end": "YYYY-MM-DD or null"
}}

Columns: {json.dumps(cols_for_prompt, default=str)}
Sample rows: {json.dumps(sample_rows[:description_sample_rows], default=str)}"""

        prompt_tokens = count_tokens(prompt, deployment)

        t = time.perf_counter()
        import openai as _openai  # noqa: PLC0415 — local import to avoid circular
        response = None
        for _attempt in range(len(retry_delays) + 1):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_completion_tokens,
                    temperature=0,
                )
                break
            except _openai.RateLimitError:
                if _attempt >= len(retry_delays):
                    raise
                _delay = retry_delays[_attempt]
                ingest_logger.warning(
                    "llm_rate_limited",
                    function="generate_file_description",
                    attempt=_attempt + 1,
                    retry_in_s=_delay,
                )
                time.sleep(_delay)
        duration = elapsed_ms(t)
        raw = response.choices[0].message.content

        api = response.usage
        p_tok = api.prompt_tokens if api else prompt_tokens
        c_tok = api.completion_tokens if api else count_tokens(raw, deployment)

        parsed = safe_parse_json(raw)
        if not parsed.get("summary"):
            parsed = {
                "summary": filename,
                "good_for": [],
                "key_metrics": [],
                "key_dimensions": [],
                "date_range_start": None,
                "date_range_end": None,
            }
        parsed["_p_tok"] = p_tok
        parsed["_c_tok"] = c_tok
        parsed["_duration"] = duration
        parsed["_deployment"] = deployment
        return parsed

    ingest_logger.info(
        "llm_call",
        function="generate_file_description",
        status="started",
        filename=filename,
        column_count=len(columns_info),
    )
    result = await asyncio.to_thread(_run)
    track_and_log(
        function="generate_file_description",
        model=result.pop("_deployment"),
        prompt_tokens=result.pop("_p_tok"),
        completion_tokens=result.pop("_c_tok"),
        duration_ms=result.pop("_duration"),
        extra={"filename": filename, "summary": result.get("summary", "")[:120]},
    )
    ingest_logger.info(
        "llm_call",
        function="generate_file_description",
        status="done",
        filename=filename,
        summary=result.get("summary", "")[:150],
        good_for=result.get("good_for", []),
    )
    return result


async def classify_query(query: str) -> dict:
    """
    Single LLM call for:
      - intent
      - entities
      - behaviors

    Returns:
    {
        "intent": str,
        "entities": list[str],
        "behaviors": list[str],
        "confidence": float,
    }

    Never raises.
    """

    def _run() -> dict:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI

        prompt = f"""
        You are a business query classifier.

        Allowed intents:
        - aggregation
        - aggregation_time_filtered
        - open_items
        - open_items_time_filtered
        - top_n_lookup
        - detail_lookup
        - complex_multi_step
        - unknown

        Allowed behaviors:
        - aggregation
        - time_filtered
        - open_items
        - top_n
        - detail_rows
        - summary
        - multi_step

        Return ONLY valid JSON.

        Schema:
        {{
        "intent": "one_allowed_intent",
        "entities": ["entity1", "entity2"],
        "behaviors": ["behavior1", "behavior2"]
        }}

        Rules:
        - entities must be business nouns.
        - normalize entity names.
        - do not return explanations.
        - do not return markdown.
        - intent MUST come from the allowed list.
        - behaviors MUST come from the allowed list.

        Query:
        {query}
        """

        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=150,
        )

        raw = (resp.choices[0].message.content or "{}").strip()
        parsed = safe_parse_json(raw)

        intent = str(parsed.get("intent", "unknown")).strip()

        entities = _clean_extracted_entities(
            parsed.get("entities", []),
            query,
        )

        behaviors = [
            str(x).strip().lower()
            for x in parsed.get("behaviors", [])
            if str(x).strip()
        ]

        if intent == "unknown":
            confidence = 0.30
        else:
            confidence = 0.80

        return {
            "intent": intent,
            "entities": entities,
            "behaviors": behaviors,
            "confidence": confidence,
        }

    try:
        return await asyncio.to_thread(_run)

    except Exception as exc:
        chat_logger.warning(
            "query_classification_error",
            error=str(exc)[:200],
        )

        return {
            "intent": "unknown",
            "entities": [],
            "behaviors": [],
            "confidence": 0.0,
        }
 



async def enrich_semantic_description(
    filename: str,
    current_description: str,
    current_good_for: list,
    role_groups: list,
    neighbors: list,
    grain: str | None = None,
) -> dict:
    """Generate additional good_for phrases using workflow signals.

    Uses same-role-kind column groups and approved relationship neighbors to
    produce workflow-aware question phrases that the schema-only Stage 4 prompt
    cannot generate. Bounded by INGEST_SEMANTIC_ENRICHMENT_MAX_ADDITIONS.

    Returns dict with key ``additional_good_for`` (list[str]).
    Never raises; returns empty list on LLM/parse failure.
    """
    def _run() -> dict:
        settings = get_settings()
        max_completion_tokens = max(
            1, int(getattr(settings, "INGEST_SEMANTIC_ENRICHMENT_MAX_COMPLETION_TOKENS", 400))
        )
        max_additions = max(
            1, int(getattr(settings, "INGEST_SEMANTIC_ENRICHMENT_MAX_ADDITIONS", 5))
        )
        client, deployment = get_client()

        groups_section = ""
        if role_groups:
            lines = [
                f"  - [{g['kind']}:{g['label']}] {', '.join(g['columns'])}"
                for g in role_groups
            ]
            groups_section = (
                "\nSame-role column groups (candidates for ratio, completion,"
                " reconciliation questions):\n" + "\n".join(lines)
            )

        neighbors_section = ""
        if neighbors:
            lines = []
            for n in neighbors:
                lines.append(
                    f"  - {n['name']} ({n['relationship_type']} via"
                    f" {n['join_column_this']} \u2192 {n['join_column_neighbor']})"
                    f"\n    Description: {n['neighbor_description']}"
                    f"\n    Used for: {'; '.join(n['neighbor_good_for'][:3])}"
                )
            neighbors_section = (
                "\nApproved join partners (data-verified):\n" + "\n".join(lines)
            )

        grain_line = f"\nTable grain: {grain}" if grain else ""

        prompt = (
            f'You are enriching the semantic catalog entry for a table named "{filename}".\n\n'
            f"Current description: {current_description}\n\n"
            f"Current questions this table supports:\n"
            + "\n".join(f"  - {q}" for q in current_good_for)
            + grain_line
            + groups_section
            + neighbors_section
            + f"\n\nTask: Generate up to {max_additions} ADDITIONAL natural language question"
            " phrases this table can support.\nFocus on:\n"
            "1. Questions comparing columns from same-role groups (ratio, completion, backlog,"
            " reconciliation).\n"
            "2. Questions requiring joining to the listed approved neighbors.\n"
            "3. Operational lifecycle questions derivable from the column structure and"
            " relationships above.\n\n"
            "Rules:\n"
            "- Do NOT repeat any existing question.\n"
            "- Do NOT use ERP-specific vocabulary (Oracle, SAP, AR, AP) unless those terms"
            " appear in the table or neighbor names above.\n"
            "- Do NOT invent relationships or columns not listed above.\n"
            "- Each phrase should be specific and answerable by a data analyst.\n"
            "- Generate ONLY questions structurally inferable from the data above.\n\n"
            'Return ONLY this JSON:\n{"additional_good_for": ["phrase 1", "phrase 2", ...]}'
        )

        raw_retry_delays = settings.INGEST_LLM_RETRY_DELAYS_SECONDS
        retry_delay_items = (
            raw_retry_delays.split(",")
            if isinstance(raw_retry_delays, str)
            else raw_retry_delays
        )
        retry_delays = [
            max(0, int(str(raw).strip()))
            for raw in retry_delay_items
            if str(raw).strip()
        ]

        t = time.perf_counter()
        import openai as _openai  # noqa: PLC0415 — local import to avoid circular
        response = None
        for _attempt in range(len(retry_delays) + 1):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=max_completion_tokens,
                    temperature=0,
                )
                break
            except _openai.RateLimitError:
                if _attempt >= len(retry_delays):
                    raise
                _delay = retry_delays[_attempt]
                ingest_logger.warning(
                    "llm_rate_limited",
                    function="enrich_semantic_description",
                    attempt=_attempt + 1,
                    retry_in_s=_delay,
                )
                time.sleep(_delay)

        duration = elapsed_ms(t)
        raw = response.choices[0].message.content
        api = response.usage
        p_tok = api.prompt_tokens if api else 0
        c_tok = api.completion_tokens if api else 0

        parsed = safe_parse_json(raw)
        additions = parsed.get("additional_good_for", [])
        if not isinstance(additions, list):
            additions = []
        parsed["additional_good_for"] = additions
        parsed["_p_tok"] = p_tok
        parsed["_c_tok"] = c_tok
        parsed["_duration"] = duration
        parsed["_deployment"] = deployment
        return parsed

    ingest_logger.info(
        "llm_call",
        function="enrich_semantic_description",
        status="started",
        filename=filename,
        role_groups=len(role_groups),
        neighbors=len(neighbors),
    )
    try:
        result = await asyncio.to_thread(_run)
    except Exception as exc:
        ingest_logger.warning(
            "llm_call",
            function="enrich_semantic_description",
            status="error",
            filename=filename,
            error=str(exc)[:200],
        )
        return {"additional_good_for": []}
    track_and_log(
        function="enrich_semantic_description",
        model=result.pop("_deployment"),
        prompt_tokens=result.pop("_p_tok"),
        completion_tokens=result.pop("_c_tok"),
        duration_ms=result.pop("_duration"),
        extra={"filename": filename},
    )
    ingest_logger.info(
        "llm_call",
        function="enrich_semantic_description",
        status="done",
        filename=filename,
        additions=len(result.get("additional_good_for", [])),
    )
    return result
