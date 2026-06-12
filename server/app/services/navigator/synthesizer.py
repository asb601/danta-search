"""[6] SYNTHESIZE — one mini call that phrases the VERIFIED numbers as prose.

The final stage of the loop. By the time we get here every number in the
``StepLedger`` was produced by DataFusion/DuckDB on Parquet (I11) and every
structural decision was value-verified (I6). mini's ONLY job here is PROSE: read
the question + the ledger's verified results and phrase them in business
language. It does the SAME mini-call discipline planner.py / proposer.py use
(temp 0, ``AZURE_OPENAI_DEPLOYMENT_MINI``, ``safe_parse_json`` is unnecessary —
we want prose, not JSON).

CRITICAL (INVARIANTS I1/I2):
  * mini does NOT compute a number and does NOT invent one. Every figure it may
    use is handed to it pre-computed in the ledger digest. The prompt forbids new
    numbers explicitly, and the safe fallback (when the LLM errors) is a TEMPLATED
    string built deterministically from the same ledger — so a number can NEVER
    originate in this module.
  * mini does NOT relabel a result, pick a table, or write SQL — those stages are
    already done and frozen in the ledger.

Robustness: NEVER raises. An LLM error, an empty response, or any exception falls
back to a deterministic templated sentence assembled from the ledger's verified
results. The driver always gets a non-empty answer string.
"""
from __future__ import annotations

import asyncio
import re
from typing import Any

import structlog

from app.core.config import get_settings
from app.core.openai_client import get_client
from app.services.navigator.types import StepLedger, StepResult

logger = structlog.get_logger("navigator.synthesizer")

# How many sample rows per step to expose to mini — enough to phrase a small
# result, capped so cost stays flat regardless of how wide a step's output is.
_DIGEST_ROWS = 5

# Number-grounding guard (FIX A / M6). A numeric literal in the prose is a VIOLATION
# only if it is "magnitude-like" — has a decimal point, a thousands separator, or an
# absolute value >= this floor — AND is not in the grounded set. Small bare integers
# ("top 5", "3 vendors", ordinals, grounded years) are NEVER violations: they are the
# structural/linguistic glue a faithful answer uses, not invented figures. This keeps
# the guard TOLERANT (no false reject on good prose) while still catching a fabricated
# figure — the one place a wrong number could surface in an otherwise number-safe pipe.
_MAGNITUDE_FLOOR = 1000.0
# Matches an integer or decimal with optional thousands separators and trailing %:
# 12,345  1234.5  98,765.43  3  2024  50%  — capturing the numeric token only.
_NUMBER_RE = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _fmt_number(value: Any) -> str:
    """Render a scalar for prose: thousands-separated for whole numbers, trimmed
    for floats. Pure — never raises (falls back to ``str``)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return str(value)
    if f == int(f):
        return f"{int(f):,}"
    return f"{f:,.4g}"


def _step_digest(result: StepResult) -> dict:
    """Project ONE executed step into a compact, number-bearing digest for the
    prompt. Carries ONLY values that came from the engine (scalar / rows / total /
    measure_label) so mini phrases, never computes. Pure."""
    digest: dict[str, Any] = {
        "step_id": result.step_id,
        "measure": result.measure_label or "result",
    }
    if result.table:
        digest["table"] = result.table
    if result.grain:
        digest["grain"] = result.grain
    if result.scalar is not None:
        digest["value"] = result.scalar
    if result.total is not None:
        digest["total_rows"] = result.total
    rows = list(result.rows or ())[:_DIGEST_ROWS]
    if rows:
        digest["rows"] = rows
    return digest


def _ledger_digest(ledger: StepLedger) -> list[dict]:
    """The full per-step digest the prompt and the fallback both read. Pure."""
    return [_step_digest(r) for r in ledger.results.values()]


def _fallback_prose(question: str, ledger: StepLedger) -> str:
    """Deterministic templated answer assembled from the ledger's VERIFIED results.

    Used when the LLM errors so the driver always returns a non-empty string. The
    numbers come straight from the ledger (I2 — never invented here)."""
    parts: list[str] = []
    for result in ledger.results.values():
        measure = result.measure_label or "result"
        table = f" over {result.table}" if result.table else ""
        if result.scalar is not None:
            parts.append(f"{measure}{table}: {_fmt_number(result.scalar)}")
        elif result.total is not None:
            shown = (
                f" (showing the top {len(result.rows)})"
                if result.total > len(result.rows)
                else ""
            )
            parts.append(f"{_fmt_number(result.total)} rows for {measure}{table}{shown}")
        else:
            parts.append(f"No result for {measure}{table}")
    if not parts:
        return "I couldn't produce a verified result for that question."
    return "; ".join(parts) + "."


def _normalize_number(token: str) -> str:
    """Normalize a numeric token to a canonical comparable form: strip thousands
    separators and a trailing percent, drop a redundant trailing ``.0``. Pure; the
    original string is returned on any parse failure so it can still match a raw form.
    """
    cleaned = token.replace(",", "").rstrip("%")
    try:
        f = float(cleaned)
    except (TypeError, ValueError):
        return token
    if f == int(f):
        return str(int(f))
    return repr(f)


def _add_grounded(grounded: set[str], value: Any) -> None:
    """Add a single numeric value to the grounded set in BOTH the formatted
    (``_fmt_number``) and raw (``str``) forms, plus its normalized form, so a prose
    literal matches regardless of how mini rendered it. Pure; non-numerics are added
    only as their raw/formatted string (harmless). Never raises."""
    grounded.add(_fmt_number(value))
    grounded.add(str(value))
    grounded.add(_normalize_number(str(value)))


def _numeric_values_in_row(row: Any) -> list[Any]:
    """Every numeric value in one result row (dict values, or the bare value of a
    scalar row). Pure; year-like ints are returned too so the synthesizer may cite a
    year that appears in the data without tripping the guard."""
    out: list[Any] = []
    if isinstance(row, dict):
        candidates = row.values()
    else:
        candidates = (row,)
    for v in candidates:
        if isinstance(v, bool):  # bool is an int subclass — never a figure
            continue
        if isinstance(v, (int, float)):
            out.append(v)
    return out


def _grounded_number_set(ledger: StepLedger) -> set[str]:
    """The set of numbers a faithful synthesis is ALLOWED to state, drawn ENTIRELY
    from the ledger (I2) — every step's scalar, total, and each numeric value in its
    rows, in formatted + raw + normalized forms. Plus STRUCTURALLY-allowed numbers:
    any ``top_n``, each step's ``len(rows)``, and year-like integers in rows. Pure."""
    grounded: set[str] = set()
    for result in ledger.results.values():
        if result.scalar is not None:
            _add_grounded(grounded, result.scalar)
        if result.total is not None:
            _add_grounded(grounded, result.total)
        rows = list(result.rows or ())
        # Structural: the count of rows is a legitimate figure ("the top 3").
        _add_grounded(grounded, len(rows))
        for row in rows:
            for v in _numeric_values_in_row(row):
                _add_grounded(grounded, v)
        # Structural: a top_n carried on the result (if present) is allowed.
        top_n = getattr(result, "top_n", None)
        if isinstance(top_n, int):
            _add_grounded(grounded, top_n)
    return grounded


def _is_magnitude_like(token: str, normalized: str) -> bool:
    """A literal is "magnitude-like" (and therefore checkable) iff it has a decimal
    point, OR a thousands separator, OR an absolute value >= ``_MAGNITUDE_FLOOR``.
    Small bare integers (counts, ordinals, years) are NOT magnitude-like. Pure."""
    if "." in token or "," in token:
        return True
    try:
        return abs(float(normalized)) >= _MAGNITUDE_FLOOR
    except (TypeError, ValueError):
        return False


def _prose_is_grounded(prose: str, ledger: StepLedger) -> bool:
    """TOLERANT number-grounding check (FIX A / M6). Returns False iff the prose
    contains a MAGNITUDE-LIKE numeric literal whose normalized form is NOT in the
    ledger-derived grounded set; True otherwise.

    Bias is toward KEEPING the LLM prose: only a positive, magnitude-like, ungrounded
    figure is a violation. Small bare integers ("top 5", "3 vendors"), ordinals, and
    grounded years never trip it, so a faithful answer is never degraded. Pure — reads
    only the prose + the ledger, never an external number source."""
    if not prose:
        return True
    grounded = _grounded_number_set(ledger)
    for match in _NUMBER_RE.finditer(prose):
        token = match.group(0).strip(",")  # a trailing list comma is not part of it
        if not token or not any(ch.isdigit() for ch in token):
            continue
        normalized = _normalize_number(token)
        # Grounded in ANY stored form (formatted / raw / normalized) -> fine.
        if token in grounded or normalized in grounded:
            continue
        # Not grounded: a violation ONLY when it is magnitude-like. Small bare ints
        # (counts, ordinals, years) are tolerated even when not explicitly grounded.
        if _is_magnitude_like(token, normalized):
            return False
    return True


def _prompt(question: str, digest: list[dict]) -> str:
    return f"""You are the SYNTHESIZE stage of a data analytics agent. The analysis is DONE:
the verified results below were computed by the query engine over the data. Your
ONLY job is to phrase them as a short, clear, business-friendly answer to the
question. You MUST NOT introduce any number that is not in the results, MUST NOT
recompute or relabel anything, and MUST NOT mention SQL, tables, or column names
unless they help the reader.

QUESTION: {question}

VERIFIED RESULTS (every number here came from the engine — use these verbatim):
{digest}

Write 1-3 sentences answering the question using ONLY the figures above. If a
result has a single ``value``, state it. If it has ``rows``, summarise them. Plain
prose only — no JSON, no markdown headers."""


async def synthesize(question: str, ledger: StepLedger) -> str:
    """ONE mini call (temp 0) → business prose around the ledger's VERIFIED numbers.

    NEVER raises and NEVER invents a number: the figures are handed to mini in the
    ledger digest; on any LLM error a deterministic templated sentence built from
    the SAME ledger is returned (I2). An empty ledger yields a safe abstain-style
    line.
    """
    if ledger is None or not ledger.results:
        return "I couldn't produce a verified result for that question."

    digest = _ledger_digest(ledger)

    def _run() -> str:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": _prompt(question, digest)}],
            temperature=0,
            max_completion_tokens=400,
        )
        return (resp.choices[0].message.content or "").strip()

    try:
        prose = await asyncio.to_thread(_run)
    except Exception as exc:  # noqa: BLE001 — never raise; templated fallback
        logger.warning("synthesize_llm_error", error=str(exc)[:200])
        return _fallback_prose(question, ledger)

    if not prose:
        logger.info("synthesize_empty_prose")
        return _fallback_prose(question, ledger)

    # Number-grounding guard (I1/I2 / M6): the ONE place a hallucinated figure could
    # otherwise surface verbatim. If the prose states a magnitude-like number not in
    # the ledger, DISCARD the LLM text and return the deterministic templated answer
    # built from the SAME verified ledger — never the ungrounded prose.
    if not _prose_is_grounded(prose, ledger):
        logger.info("synthesize_ungrounded_number", n_steps=len(ledger.results))
        return _fallback_prose(question, ledger)

    logger.info("synthesize_ok", n_steps=len(ledger.results), chars=len(prose))
    return prose
