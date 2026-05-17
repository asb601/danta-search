"""Column semantic role resolver — LLM-only, no heuristics.

WHAT THIS DOES
==============
Every column in every ingested file gets a semantic role assigned once at
ingest time. The result is stored in FileMetadata.column_semantic_roles (JSONB)
and never re-computed unless the file is re-ingested.

WHY LLM-ONLY (no heuristics)
=============================
Heuristics require a maintained vocabulary of known column aliases.
That vocabulary breaks on every new client system because each source system
uses different codes and naming conventions.
There is no finite alias list that covers all naming conventions.

The LLM receives column names + data types + sample values in one call and
returns the full role map. It sees the actual data, not just the column name.
Cost: one API call per file at ingest. Never called again for that file.

ROLE REGISTRY
=============
Code defines behavior kinds, not a fixed business role list. Tenant config may
provide known roles. Otherwise the LLM returns a typed dynamic role:
custom:<kind>:<slug>. The kind gives downstream code safe behavior without
hardcoding every future domain noun.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from app.core.logger import ingest_logger
from app.services.semantic_roles import (
    ROLE_KINDS,
    is_dynamic_role,
    is_valid_role,
    role_definitions_for_prompt,
    valid_roles,
)


async def resolve_column_roles(
    columns_info: list[dict[str, Any]],
    filename: str,
    glossary: dict[str, str] | None = None,
    semantic_config: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str]:
    """Resolve all columns to semantic roles via one LLM call.

    Sends all columns with their data type and up to 10 unique sample values.
    If a schema.csv glossary is available it is included as context so the LLM
    can understand opaque source-system codes without guessing.

    Returns:
        roles  : {col_name: role_string} — only columns where role was resolved.
                 Columns the LLM cannot confidently classify are excluded (clean map).
        source : "llm" always.
    """
    if not columns_info:
        return {}, "llm"

    roles = await _call_llm(columns_info, filename, glossary, semantic_config)
    source = "llm_dynamic" if any(is_dynamic_role(role) for role in roles.values()) else "llm"

    ingest_logger.info(
        "column_role_resolver",
        filename=filename,
        total_columns=len(columns_info),
        resolved=len(roles),
        coverage_pct=round(len(roles) / len(columns_info) * 100, 1) if columns_info else 0,
        roles_preview={k: v for k, v in list(roles.items())[:10]},
    )

    return roles, source


async def _call_llm(
    columns_info: list[dict[str, Any]],
    filename: str,
    glossary: dict[str, str] | None,
    semantic_config: dict[str, Any] | None,
) -> dict[str, str]:
    """Single LLM call — classifies ALL columns at once with full data profile."""
    from app.core.openai_client import get_client  # local to avoid circular
    from app.core.token_counter import count_tokens, elapsed_ms, track_and_log

    # Build column profile: name + type + up to 10 unique non-null sample values
    col_profiles = []
    for c in columns_info[:60]:  # cap at 60 columns to stay within token budget
        name = c.get("name", "")
        if not name:
            continue
        samples = c.get("unique_values") or c.get("sample_values") or []
        samples = [str(v) for v in samples if v not in ("", None, "None")][:10]
        col_profiles.append({
            "name": name,
            "type": c.get("type", ""),
            "samples": samples,
        })

    # Glossary section — translates opaque ERP codes for the LLM
    glossary_section = ""
    if glossary:
        lines = [f"  {code} = {meaning}" for code, meaning in list(glossary.items())[:80]]
        glossary_section = (
            "\nSchema glossary (admin-provided mapping of column codes to business meanings):\n"
            + "\n".join(lines)
            + "\n"
        )

    allowed_roles = sorted(valid_roles(semantic_config))
    allowed_kinds = ", ".join(ROLE_KINDS)

    known_role_section = (
        json.dumps(allowed_roles)
        if allowed_roles
        else "[]  # no tenant-defined roles configured; use typed dynamic roles"
    )
    role_definition_section = role_definitions_for_prompt(semantic_config) or "No tenant-defined role definitions."

    prompt = f"""You are a data catalog expert. Classify each column into exactly one semantic role.

File: {filename}{glossary_section}
Columns (name, data type, sample values):
{json.dumps(col_profiles, indent=2)}

Known roles:
{known_role_section}

Role definitions:
{role_definition_section}

If no known role fits but the column has clear business meaning, create a dynamic role:
custom:<kind>:<short_snake_case_label>

Allowed dynamic kinds: {allowed_kinds}
Examples:
- custom:entity_key:claim identifies a claim record.
- custom:reference_key:policy_type identifies a reusable reference code.
- custom:additive_measure:premium is safe to sum.
- custom:non_additive_measure:exchange_rate is numeric but not safe to sum.
- custom:date:service_date is a date field.
- custom:attribute:coverage_tier is a descriptive grouping/filter field.

Rules:
- Return ONLY a JSON object: {{"column_name": "role", ...}}
- Prefer known roles when they fit.
- Use a dynamic role when known roles do not fit and samples/glossary make the meaning clear.
- Omit any column you cannot confidently classify — do NOT include unknown entries.
- For name/label/description/text columns, use custom:attribute:<business_label> unless a known role fits.
- For date/time/timestamp columns, use custom:date:<business_label> unless a known role fits.
- For identifiers, use custom:entity_key:<entity_label> when it identifies this file's main business entity, otherwise custom:reference_key:<reference_label>.
- For numeric facts safe to sum, use custom:additive_measure:<measure_label>.
- For numeric rates, percentages, ratios, balances, unit prices, or snapshots that are not safe to sum, use custom:non_additive_measure:<measure_label>.
- Do not add any explanation or markdown fences.
"""

    def _run() -> dict[str, str]:
        client, deployment = get_client()
        prompt_tokens = count_tokens(prompt, deployment)
        t = time.perf_counter()

        import openai as _openai  # noqa: PLC0415
        _RETRY = [1, 5, 30]
        response = None
        for attempt in range(len(_RETRY) + 1):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=600,
                    temperature=0,
                )
                break
            except _openai.RateLimitError:
                if attempt >= len(_RETRY):
                    raise
                time.sleep(_RETRY[attempt])

        duration = elapsed_ms(t)
        raw = response.choices[0].message.content if response else "{}"
        api = response.usage if response else None
        p_tok = api.prompt_tokens if api else prompt_tokens
        c_tok = api.completion_tokens if api else 0

        # Strip markdown fences if the model added them despite instructions
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = [ln for ln in cleaned.split("\n") if not ln.strip().startswith("```")]
            cleaned = "\n".join(lines).strip()

        try:
            parsed: dict = json.loads(cleaned)
        except (json.JSONDecodeError, ValueError):
            parsed = {}

        filtered = {
            k: v
            for k, v in parsed.items()
            if isinstance(k, str) and isinstance(v, str) and is_valid_role(v, semantic_config)
        }

        track_and_log(
            function="resolve_column_roles",
            model=deployment,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=duration,
            extra={"filename": filename, "resolved": len(filtered), "total": len(col_profiles)},
        )
        return filtered

    ingest_logger.info(
        "column_role_resolver",
        status="started",
        filename=filename,
        column_count=len(col_profiles),
        has_glossary=bool(glossary),
    )

    try:
        roles = await asyncio.to_thread(_run)
    except Exception as exc:
        ingest_logger.warning(
            "column_role_resolver",
            status="failed",
            filename=filename,
            error=str(exc)[:300],
        )
        return {}

    return roles
