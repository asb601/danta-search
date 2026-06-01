"""High-level LLM tasks used by backend services."""
from __future__ import annotations

import asyncio
import json
import re
import time

from app.core.config import get_settings
from app.core.logger import chat_logger, ingest_logger
from app.core.openai_client import chat_complete_with_failover, get_chat_client, get_client
from app.core.token_counter import count_tokens, elapsed_ms, track_and_log


_ENTITY_TOKEN_RE = re.compile(r"[^a-z0-9]+")
_MAX_EXTRACTED_ENTITIES = 10
_ALLOWED_INTENTS = frozenset({
    "aggregation",
    "aggregation_time_filtered",
    "open_items",
    "open_items_time_filtered",
    "top_n_lookup",
    "detail_lookup",
    "complex_multi_step",
    "unknown",
})
_ALLOWED_BEHAVIORS = frozenset({
    "aggregation",
    "time_filtered",
    "open_items",
    "top_n",
    "detail_rows",
    "summary",
    "multi_step",
})
def _entity_tokens(text: str) -> list[str]:
    return [t for t in _ENTITY_TOKEN_RE.split(text.lower()) if t]


def _entity_text(raw: object) -> str:
    if isinstance(raw, dict):
        for key in ("name", "entity", "concept", "label", "source_phrase"):
            value = raw.get(key)
            if str(value or "").strip():
                return str(value)
        return ""
    return str(raw or "")


def _bounded_confidence(value: object, fallback: float) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        confidence = fallback
    return max(0.0, min(1.0, confidence))


def _as_list(value: object) -> list:
    if isinstance(value, list):
        return value
    if value is None:
        return []
    return [value]


def _clean_extracted_entities(entities: list, query: str = "") -> list[str]:
    """Normalize LLM-selected concepts without semantic filtering."""
    cleaned: list[str] = []
    seen: set[str] = set()
    for raw in entities:
        tokens = _entity_tokens(_entity_text(raw))
        if not tokens:
            continue
        entity = "_".join(tokens)
        if entity and entity not in seen:
            cleaned.append(entity)
            seen.add(entity)
    return cleaned[:_MAX_EXTRACTED_ENTITIES]


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
    tier: str = "standard",
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
        _, deployment = get_chat_client(tier=tier)
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
                response = chat_complete_with_failover(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
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
        You are the query decomposition stage of a business analytics agent.
        Your output drives table retrieval and query planning — choose entities that
        tell the planner WHAT data objects to look for, not HOW to present results.

        Entities are business objects, processes, workflow/lifecycle states, exceptions,
        and relationships that require distinct data access (different tables, joins,
        filters, or exception records). They are NOT display fields, time ranges,
        metrics, output instructions, or narrative requests.

        Allowed intents: aggregation, aggregation_time_filtered, open_items,
        open_items_time_filtered, top_n_lookup, detail_lookup, complex_multi_step, unknown

        Allowed behaviors: aggregation, time_filtered, open_items, top_n,
        detail_rows, summary, multi_step

        Return ONLY valid JSON:
        {{
          "intent": "one_allowed_intent",
          "entities": ["snake_case_concept"],
          "behaviors": ["behavior"],
          "confidence": 0.0
        }}

        Rules:
        1. Expand abbreviations (PO → purchase_order, SO → sales_order, etc.).
        2. Include the primary object plus any workflow states, exceptions, lifecycle
           events, matching/reconciliation concepts, or holds that need separate data.
        3. Bullet lists and sections after "summarize:", "including:", etc. are
           analysis components — judge each one: keep if it needs its own data,
           skip if it's just a display field or a requested answer section.
        4. Generic labels (status, approval, issue, hold, delay) must be anchored:
           write po_approval_status not approval_status, invoice_mismatch not mismatch.
        5. Exclude: time ranges, metrics/values, display-only attributes, recommendations,
           next actions, narratives, and summaries.
        6. Prefer breadth — do not collapse a multi-component query to one entity.
        7. Add time_filtered to behaviors when the query mentions any time period.

        Query: {query}
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
            max_completion_tokens=500,
        )

        raw = (resp.choices[0].message.content or "{}").strip()
        parsed = safe_parse_json(raw)

        intent = str(parsed.get("intent", "unknown")).strip().lower()
        if intent not in _ALLOWED_INTENTS:
            intent = "unknown"

        entities = _clean_extracted_entities(
            _as_list(parsed.get("entities")),
            query,
        )

        behaviors = [
            str(x).strip().lower()
            for x in _as_list(parsed.get("behaviors"))
            if str(x).strip().lower() in _ALLOWED_BEHAVIORS
        ]

        fallback_confidence = 0.30 if intent == "unknown" else (0.75 if entities else 0.50)
        confidence = _bounded_confidence(parsed.get("confidence"), fallback_confidence)
        if intent == "unknown":
            confidence = min(confidence, 0.45)

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
    tier: str = "standard",
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
        _, deployment = get_chat_client(tier=tier)

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
                response = chat_complete_with_failover(
                    messages=[{"role": "user", "content": prompt}],
                    tier=tier,
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
