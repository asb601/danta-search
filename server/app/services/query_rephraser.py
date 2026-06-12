"""Query Rephrasing Layer — turn a chat prompt into a precise ERP analytical request.

Sits at the chat entry point, BEFORE run_agent_query_stream(). One LLM call
rewrites the user's question into ONE compact, machine-style retrieval directive
using Oracle OEBS / SAP ECC domain expertise — naming the source tables, the join
keys/columns, the output columns (with readable aliases, coded foreign keys
resolved via lookup tables), and the precise filter conditions. The output is fed
straight into the query pipeline, so it is directive-only: no preamble, caveats,
explanation, or markdown (any extra prose would corrupt the downstream request).

Bounds (non-negotiable):
  - It REPHRASES, it does not answer. Plain-text in, plain-text out. No tools,
    no SQL, no DB reads. The cleaned text fully replaces the original downstream.
  - It has NO file/schema access, so it cannot itself verify that a column or
    table exists — it expresses the request precisely in standard ERP terms and
    the DOWNSTREAM agent (which has the catalog + column tools) verifies every
    table/column before use. The rephraser is upstream of, and never bypasses,
    that VERIFY gate.
  - Never raises. Any problem (disabled flag, LLM error, empty / runaway output)
    falls back to the ORIGINAL query, so a bad rephrase degrades to today's
    behavior rather than corrupting the ask.
  - Flag-gated by QUERY_REPHRASE_ENABLED. Chat only — dashboards call the agent
    with machine-clean intents and are not routed through here.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logger import chat_logger
from app.core.openai_client import get_client


# Runaway guard — the rewrite is ONE compact directive (tables/joins/columns/
# filters), so it stays short even for a multi-part question. Anything past this
# means the model ignored the "no extra text" rule and started explaining or
# answering — reject it and keep the original.
_MAX_REPHRASE_CHARS = 3000

# Token ceiling for the rewrite. The visible output is short, but a gpt-5.x mini
# may also spend completion tokens on reasoning before emitting it, so keep
# headroom to avoid truncating the directive mid-sentence.
_REPHRASE_MAX_TOKENS = 1500


def _is_runaway(rephrased: str) -> bool:
    return len(rephrased) > _MAX_REPHRASE_CHARS


def _append_domain(text: str, domain: str | None) -> str:
    """Append the active domain marker ``-- <domain>`` to the directive.

    The domain is NOT inferred by the LLM — it is supplied by the caller from the
    request's domain tag / selected domain filter and appended deterministically
    here. Idempotent: if the marker is already present, the text is unchanged.
    """
    if not domain:
        return text
    d = str(domain).strip()
    if not d:
        return text
    marker = f"-- {d}"
    if marker.lower() in text.lower():
        return text
    return f"{text.rstrip()} {marker}"


@dataclass
class RephraseResult:
    """Stable output of the rephraser.

    text    — the query to send downstream (rephrased, or the original on fallback)
    changed — True only when text differs from the input
    reason  — short tag: "rephrased" | "unchanged" | "disabled" | "empty_input"
              | "error" | "empty" | "too_long"
    """
    text: str
    changed: bool
    reason: str


REPHRASE_PROMPT = """You are an elite enterprise ERP and data-warehouse analyst.{domain_block}

Rewrite the user's question as ONE precise, compact data-retrieval directive that \
a downstream query engine will execute. Using your ERP schema knowledge, name the \
exact source tables (as many as the question requires), the join keys between them, \
the output columns with readable aliases (resolving coded foreign keys via the \
correct lookup tables), and the precise filter conditions the question implies.

HARD RULES — the output is fed directly into a query pipeline, so ANY extra text \
breaks it:
- Output ONLY the rewritten directive. No preamble, no explanation, no notes, no \
caveats, no "verify columns" remarks, no data-quality commentary, no headings, no \
markdown, no quotes, no bullet points, no trailing notes.
- Do NOT answer the question or invent data values.
- Do NOT add or drop any condition the user actually asked for.
- Use as many tables and joins as the question genuinely requires — do not \
artificially limit to fewer tables than needed.{context_block}

Question: {query}
Rewrite:"""


async def rephrase_query(
    query: str,
    *,
    conversation_context: str = "",
    domain: str | None = None,
    log_context: dict | None = None,
) -> RephraseResult:
    """Return an ERP-precise rewrite of ``query`` (or the original on any fallback).

    ``conversation_context`` lets the model resolve a dangling reference (a
    pronoun, "the same period") so the rewritten request stands alone.
    ``domain`` is the active domain tag / selected domain filter from the request.
    When set it is (1) injected into the prompt so the model targets that system's
    tables, and (2) appended to the rewrite as ``-- <domain>`` (in code). The domain
    value comes from the request, never inferred by the LLM. ``log_context`` is
    merged into the structured ``query_rephrased`` event.
    """
    settings = get_settings()
    extra = log_context or {}

    if not getattr(settings, "QUERY_REPHRASE_ENABLED", False):
        return RephraseResult(query, False, "disabled")

    original = query.strip()
    if not original:
        return RephraseResult(query, False, "empty_input")

    def _run() -> str:
        client, _ = get_client()
        # Dedicated rephrase deployment (e.g. a gpt-5.4-mini deployment) when set;
        # otherwise reuse the standard mini deployment. Same endpoint/key client.
        deployment = (
            getattr(settings, "QUERY_REPHRASE_DEPLOYMENT", "")
            or settings.AZURE_OPENAI_DEPLOYMENT_MINI
        )

        # Inject the active domain (from the selected domain filter) so the model
        # targets THAT system's tables — e.g. SAP ECC tables for a SAP dataset
        # rather than defaulting to Oracle tables on Oracle-sounding phrasing.
        domain_block = ""
        if domain and str(domain).strip():
            d = str(domain).strip()
            domain_block = (
                f"\n\nThe target source system / dataset is: {d}. Name ONLY tables "
                f"and columns that belong to {d}; never use tables from any other "
                f"ERP system, even if the question's wording resembles one."
            )

        context_block = ""
        if conversation_context.strip():
            context_block = (
                "\n\nRecent conversation (use ONLY to resolve a dangling reference "
                "such as a pronoun — do not import other facts):\n"
                + conversation_context.strip()[:2000]
            )

        prompt = REPHRASE_PROMPT.format(
            query=original, context_block=context_block, domain_block=domain_block
        )
        messages = [{"role": "user", "content": prompt}]

        try:
            resp = client.chat.completions.create(
                model=deployment,
                messages=messages,
                temperature=0,
                max_completion_tokens=_REPHRASE_MAX_TOKENS,
            )
        except Exception as exc:  # noqa: BLE001
            # Some newer deployments (gpt-5 family) only allow the default
            # temperature and 400 on temperature=0 — retry without it.
            if "temperature" in str(exc).lower():
                resp = client.chat.completions.create(
                    model=deployment,
                    messages=messages,
                    max_completion_tokens=_REPHRASE_MAX_TOKENS,
                )
            else:
                raise
        return (resp.choices[0].message.content or "").strip()

    try:
        rephrased = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001 — a rephrase failure must never break chat
        chat_logger.warning("query_rephrase_error", error=str(exc)[:200], **extra)
        return RephraseResult(query, False, "error")

    if not rephrased:
        return RephraseResult(query, False, "empty")

    if _is_runaway(rephrased):
        chat_logger.info(
            "query_rephrased",
            original=original[:300],
            rephrased=rephrased[:300],
            changed=False,
            reason="too_long",
            **extra,
        )
        return RephraseResult(query, False, "too_long")

    final = _append_domain(rephrased, domain)
    changed = final != original
    reason = "rephrased" if changed else "unchanged"
    chat_logger.info(
        "query_rephrased",
        original=original[:300],
        rephrased=final[:500],
        changed=changed,
        reason=reason,
        **extra,
    )
    return RephraseResult(final, changed, reason)
