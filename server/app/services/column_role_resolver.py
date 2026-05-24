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

PHASE 5: ROLE CONFIDENCE EVIDENCE
==================================
The LLM now returns per-column evidence alongside each role assignment:
  {"col": {"role": "custom:entity_key:vendor", "confidence": 0.92,
           "signals": ["column_name", "value_pattern"]}}

Valid signals the LLM may report:
  "column_name"   — column name pattern strongly matches the role kind
  "value_pattern" — sample values exhibit patterns consistent with the role
  "data_type"     — declared data type aligns with the role
  "glossary"      — a glossary entry confirmed the classification

confidence >= 0.50 is required for inclusion. Below that the LLM should omit
the column (same rule as before: no uncertain entries).

The evidence dict is stored separately in FileMetadata.column_role_evidence.
The flat role map in FileMetadata.column_semantic_roles is unchanged so all
downstream readers remain unaffected.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from app.core.config import get_settings
from app.core.logger import ingest_logger
from app.services.ingestion_config import null_tokens_lower
from app.services.semantic_roles import (
    ROLE_KINDS,
    is_dynamic_role,
    is_valid_role,
    role_definitions_for_prompt,
    valid_roles,
)

# Valid signal labels the LLM may report — bounded set prevents garbage signals
_VALID_SIGNALS: frozenset[str] = frozenset({"column_name", "value_pattern", "data_type", "glossary"})


async def resolve_column_roles(
    columns_info: list[dict[str, Any]],
    filename: str,
    glossary: dict[str, str] | None = None,
    semantic_config: dict[str, Any] | None = None,
) -> tuple[dict[str, str], str, dict[str, dict]]:
    """Resolve all columns to semantic roles via one LLM call.

    Sends all columns with their data type and up to 10 unique sample values.
    If a schema.csv glossary is available it is included as context so the LLM
    can understand opaque source-system codes without guessing.

    Returns:
        roles    : {col_name: role_string} — only columns where role was resolved
                   with confidence >= 0.50. Backward-compatible flat map.
        source   : "llm" | "llm_dynamic"
        evidence : {col_name: {"confidence": float, "signals": [str], "source": str}}
                   Per-column confidence and supporting signals. Empty dict if the
                   LLM returned old flat format (graceful degradation).
    """
    if not columns_info:
        return {}, "llm", {}

    settings = get_settings()
    preview_items = max(0, int(settings.INGEST_ROLE_RESOLVER_PREVIEW_ITEMS))
    roles, evidence = await _call_llm(columns_info, filename, glossary, semantic_config)
    source = "llm_dynamic" if any(is_dynamic_role(role) for role in roles.values()) else "llm"

    ingest_logger.info(
        "column_role_resolver",
        filename=filename,
        total_columns=len(columns_info),
        resolved=len(roles),
        coverage_pct=round(len(roles) / len(columns_info) * 100, 1) if columns_info else 0,
        roles_preview={k: v for k, v in list(roles.items())[:preview_items]},
        avg_confidence=round(
            sum(e.get("confidence", 0.0) for e in evidence.values()) / len(evidence), 3
        ) if evidence else None,
    )

    return roles, source, evidence


async def _call_llm(
    columns_info: list[dict[str, Any]],
    filename: str,
    glossary: dict[str, str] | None,
    semantic_config: dict[str, Any] | None,
) -> tuple[dict[str, str], dict[str, dict]]:
    """Single LLM call — classifies ALL columns at once with full data profile.

    Returns (roles, evidence) where:
      roles    = {col: role_str}           — flat map for backward compatibility
      evidence = {col: {confidence, signals, source}} — per-column trust metadata
    """
    from app.core.openai_client import get_client  # local to avoid circular
    from app.core.token_counter import count_tokens, elapsed_ms, track_and_log

    settings = get_settings()
    max_columns = max(1, int(settings.INGEST_ROLE_RESOLVER_MAX_COLUMNS))
    sample_values = max(0, int(settings.INGEST_ROLE_RESOLVER_SAMPLE_VALUES))
    glossary_items = max(0, int(settings.INGEST_ROLE_RESOLVER_GLOSSARY_ITEMS))
    max_completion_tokens = max(600, int(settings.INGEST_ROLE_RESOLVER_MAX_COMPLETION_TOKENS))
    null_values = null_tokens_lower()

    col_profiles = []
    for c in columns_info[:max_columns]:
        name = c.get("name", "")
        if not name:
            continue
        samples = c.get("unique_values") or c.get("sample_values") or []
        samples = [str(v) for v in samples if v is not None and str(v).strip().lower() not in null_values][:sample_values]
        col_profiles.append({
            "name": name,
            "type": c.get("type", ""),
            "samples": samples,
        })

    # Glossary section — translates opaque ERP codes for the LLM
    glossary_section = ""
    if glossary:
        lines = [f"  {code} = {meaning}" for code, meaning in list(glossary.items())[:glossary_items]]
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
- custom:entity_key:record identifies the row grain.
- custom:reference_key:record_type identifies a reusable reference code.
- custom:additive_measure:amount is safe to sum.
- custom:non_additive_measure:rate is numeric but not safe to sum.
- custom:date:event_date is a date field.
- custom:attribute:category is a descriptive grouping/filter field.

Rules:
- Return ONLY a JSON object where each key is a column name and the value is an object with:
    "role": the semantic role string
    "confidence": a float 0.50–1.00 representing your certainty (omit column if < 0.50)
    "signals": array of evidence you used — only from: ["column_name", "value_pattern", "data_type", "glossary"]
- Example output:
  {{"vendor_id": {{"role": "custom:entity_key:vendor", "confidence": 0.95, "signals": ["column_name", "value_pattern"]}},
    "order_date": {{"role": "custom:date:order_date", "confidence": 0.90, "signals": ["data_type", "column_name"]}}}}
- Prefer known roles when they fit.
- Use a dynamic role when known roles do not fit and samples/glossary make the meaning clear.
- Omit any column you cannot confidently classify (confidence would be < 0.50) — do NOT include uncertain entries.
- For name/label/description/text columns, use custom:attribute:<business_label> unless a known role fits.
- For date/time/timestamp columns, use custom:date:<business_label> unless a known role fits.
- For identifiers, use custom:entity_key:<entity_label> when it identifies this file's main business entity, otherwise custom:reference_key:<reference_label>.
- For numeric facts safe to sum, use custom:additive_measure:<measure_label>.
- For numeric rates, percentages, ratios, balances, unit prices, or snapshots that are not safe to sum, use custom:non_additive_measure:<measure_label>.
- Do not add any explanation or markdown fences.
"""

    def _run() -> tuple[dict[str, str], dict[str, dict]]:
        client, deployment = get_client()
        prompt_tokens = count_tokens(prompt, deployment)
        # Larger budget: enriched format is ~2.5× larger than flat role strings
        completion_budget = max(600, min(max_completion_tokens, len(col_profiles) * 100))
        t = time.perf_counter()

        import openai as _openai  # noqa: PLC0415
        _RETRY = [1, 5, 30]
        response = None
        for attempt in range(len(_RETRY) + 1):
            try:
                response = client.chat.completions.create(
                    model=deployment,
                    messages=[{"role": "user", "content": prompt}],
                    max_completion_tokens=completion_budget,
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

        roles: dict[str, str] = {}
        evidence: dict[str, dict] = {}

        for col_name, val in parsed.items():
            if not isinstance(col_name, str):
                continue

            if isinstance(val, dict):
                # Phase 5 enriched format: {role, confidence, signals}
                role = val.get("role", "")
                try:
                    confidence = float(val.get("confidence", 0.0))
                except (TypeError, ValueError):
                    confidence = 0.0
                raw_signals = val.get("signals") or []
                signals = [s for s in raw_signals if isinstance(s, str) and s in _VALID_SIGNALS]

                if isinstance(role, str) and is_valid_role(role, semantic_config) and confidence >= 0.50:
                    roles[col_name] = role
                    evidence[col_name] = {
                        "confidence": round(min(1.0, max(0.0, confidence)), 3),
                        "signals": signals,
                        "source": "llm",
                    }

            elif isinstance(val, str) and is_valid_role(val, semantic_config):
                # Graceful fallback: LLM returned old flat {col: role} format
                roles[col_name] = val
                evidence[col_name] = {
                    "confidence": 0.70,   # assume moderate confidence for legacy format
                    "signals": [],
                    "source": "llm",
                }

        track_and_log(
            function="resolve_column_roles",
            model=deployment,
            prompt_tokens=p_tok,
            completion_tokens=c_tok,
            duration_ms=duration,
            extra={
                "filename": filename,
                "resolved": len(roles),
                "total": len(col_profiles),
                "completion_budget": completion_budget,
            },
        )
        return roles, evidence

    ingest_logger.info(
        "column_role_resolver",
        status="started",
        filename=filename,
        column_count=len(col_profiles),
        has_glossary=bool(glossary),
    )

    try:
        roles, evidence = await asyncio.to_thread(_run)
    except Exception as exc:
        ingest_logger.warning(
            "column_role_resolver",
            status="failed",
            filename=filename,
            error=str(exc)[:300],
        )
        return {}, {}

    return roles, evidence
