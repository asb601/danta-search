"""High-level LLM tasks used by backend services."""
from __future__ import annotations

import asyncio
import json
import time

from app.core.logger import ingest_logger
from app.core.openai_client import get_client
from app.core.token_counter import count_tokens, elapsed_ms, track_and_log


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
                uncovered_preview = ", ".join(uncovered[:20])
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
    "summary": "2 sentences: (1) what specific business questions this file can support, naming the exact columns that make it useful (e.g. 'AMOUNT_DUE_REMAINING tracks open balance', 'STATUS=OP filters open items'); (2) what this file contains that similar-sounding files do not have. Avoid claiming this is the only, best, or primary source unless that is explicitly present in metadata.",
    "good_for": ["3-6 exact natural language business question phrases this file can support. Use the actual business domain terms, column names, and metrics specific to this file — do NOT use generic placeholder terms. Make each phrase discriminative: it should clearly distinguish this file from other similar-sounding files in the catalog without claiming exclusivity."],
  "key_metrics": ["numeric columns used for aggregation and SUM/AVG calculations"],
  "key_dimensions": ["categorical, status, and ID columns used for filtering and grouping — include their important values where relevant, e.g. STATUS: OP=open CL=closed"],
  "date_range_start": "YYYY-MM-DD or null",
  "date_range_end": "YYYY-MM-DD or null"
}}

Columns: {json.dumps(cols_for_prompt, default=str)}
Sample rows: {json.dumps(sample_rows[:20], default=str)}"""

        prompt_tokens = count_tokens(prompt, deployment)

        t = time.perf_counter()
        import openai as _openai  # noqa: PLC0415 — local import to avoid circular
        _RETRY_DELAYS = [1, 5, 30]
        response = None
        for _attempt in range(len(_RETRY_DELAYS) + 1):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=600,
                    temperature=0,
                )
                break
            except _openai.RateLimitError:
                if _attempt >= len(_RETRY_DELAYS):
                    raise
                _delay = _RETRY_DELAYS[_attempt]
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
