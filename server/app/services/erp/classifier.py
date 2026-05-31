"""Data-driven ERP classification.

Given the evidence already produced during ingestion (columns, sample values,
the AI description, and detected semantic roles), infer the business-context
facts a retrieval pipeline cannot see:

    source_system    — which ERP/system of record (free-form, evidence-named)
    erp_module       — which functional module (free-form)
    domain_polarity  — customer | vendor | neutral  (universal ledger axis)
    process_role     — position on a business process (free-form slug)
    grain            — one row represents what

The classifier is intentionally NOT a dictionary of table-name patterns. It
asks an LLM to NAME what it sees from evidence, then corroborates the most
decision-critical axis (polarity) against the semantic roles already detected
at ingestion. Disagreement lowers confidence rather than silently overriding.

Everything degrades safely: any failure, or confidence below the configured
floor, yields an ``unknown``/``neutral`` classification that downstream code
treats as "no business-layer signal" — i.e. exactly today's behaviour.
"""
from __future__ import annotations

import asyncio
import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Literal

from app.core.config import get_settings
from app.core.logger import ingest_logger

Polarity = Literal["customer", "vendor", "neutral"]

# The ONLY closed vocabulary in this module. Polarity is a universal
# double-entry-accounting axis (who owes whom), not an ERP-specific heuristic:
# every business system distinguishes money-in (customer/AR/sales) from
# money-out (vendor/AP/procurement). "neutral" covers master data, ledger,
# inventory, HR, and anything the evidence does not place on one side.
_VALID_POLARITIES: frozenset[str] = frozenset({"customer", "vendor", "neutral"})

# Confidence below this floor is treated as "unknown" everywhere downstream.
_DEFAULT_CONFIDENCE_FLOOR = 0.55

# Bound the evidence we send to the LLM so cost stays flat regardless of how
# wide the source file is. These are size guards, not business logic.
_MAX_COLUMNS_IN_PROMPT = 60
_MAX_SAMPLE_ROWS_IN_PROMPT = 3
_MAX_DESCRIPTION_CHARS = 800


@dataclass
class ErpClassification:
    """One file's business-context classification, with provenance.

    ``source`` records how the classification was produced:
      "llm"            — model inference (the normal path)
      "human_override" — edited by an admin in the enrich loop
      "unknown"        — could not classify / below confidence floor (safe default)
    """

    source_system: str = "Unknown"
    erp_module: str = "Unknown"
    domain_polarity: Polarity = "neutral"
    process_role: str = "unknown"
    grain: str = ""
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    source: str = "unknown"
    model_version: str = ""

    @property
    def is_reliable(self) -> bool:
        """True only when this classification is trustworthy enough to drive a
        scoping/feasibility decision. Below this, callers must degrade to
        today's behaviour rather than act on the classification."""
        if self.source == "human_override":
            return True
        if self.source != "llm":
            return False
        return self.confidence >= _confidence_floor() and self.source_system != "Unknown"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def unknown(cls, reason: str = "") -> "ErpClassification":
        ev = [reason] if reason else []
        return cls(source="unknown", evidence=ev)


def _confidence_floor() -> float:
    return float(getattr(get_settings(), "ERP_CLASSIFICATION_CONFIDENCE_FLOOR", _DEFAULT_CONFIDENCE_FLOOR))


# ── Evidence assembly ─────────────────────────────────────────────────────────

def _column_lines(columns_info: list) -> list[str]:
    """Render columns as compact 'name (type): sample, sample' lines."""
    lines: list[str] = []
    for col in (columns_info or [])[:_MAX_COLUMNS_IN_PROMPT]:
        if isinstance(col, dict):
            name = str(col.get("name") or col.get("column") or "").strip()
            if not name:
                continue
            ctype = str(col.get("type") or col.get("dtype") or "").strip()
            samples = col.get("sample_values") or col.get("samples") or []
            sample_str = ", ".join(str(s) for s in list(samples)[:4])
            piece = f"- {name}"
            if ctype:
                piece += f" ({ctype})"
            if sample_str:
                piece += f": {sample_str}"
            lines.append(piece)
        elif isinstance(col, str):
            lines.append(f"- {col}")
    return lines


def _role_polarity_signal(column_semantic_roles: dict | None) -> tuple[Polarity, list[str]]:
    """Derive a polarity hint from semantic roles ALREADY detected at ingestion.

    This is corroboration, NOT the classifier: the LLM (which reads values and
    is language-agnostic) makes the polarity call; this only adjusts confidence.
    It scans the role *labels* the ontology stage already produced for buy-side
    vs sell-side concept tokens. The token list below is English and therefore a
    best-effort second opinion: for non-English role labels it simply returns
    'neutral' (no corroboration), which never changes the LLM's decision — it
    only forgoes the confidence boost/penalty. Returns ('neutral', []) on no signal.
    """
    if not column_semantic_roles or not isinstance(column_semantic_roles, dict):
        return "neutral", []

    # Concept tokens that, when they appear inside an already-assigned role
    # label, indicate ledger side. Kept tiny and used ONLY to corroborate the
    # LLM — never as a standalone classifier.
    customer_tokens = ("customer", "client", "sold_to", "bill_to", "ship_to", "payer", "sales", "receivable", "ar_")
    vendor_tokens = ("vendor", "supplier", "payable", "ap_", "purchas", "procure", "creditor")

    cust_hits: list[str] = []
    vend_hits: list[str] = []
    for col, role in column_semantic_roles.items():
        label = str(role or "").lower()
        if any(tok in label for tok in customer_tokens):
            cust_hits.append(f"{col}->{role}")
        if any(tok in label for tok in vendor_tokens):
            vend_hits.append(f"{col}->{role}")

    if cust_hits and not vend_hits:
        return "customer", cust_hits[:4]
    if vend_hits and not cust_hits:
        return "vendor", vend_hits[:4]
    return "neutral", (cust_hits + vend_hits)[:4]


def _build_prompt(
    *,
    filename: str,
    columns_info: list,
    sample_rows: list,
    ai_description: str | None,
    column_semantic_roles: dict | None,
) -> str:
    col_block = "\n".join(_column_lines(columns_info)) or "(no column metadata)"
    desc = (ai_description or "").strip()[:_MAX_DESCRIPTION_CHARS] or "(none)"

    sample_block = ""
    if sample_rows:
        try:
            sample_block = json.dumps(sample_rows[:_MAX_SAMPLE_ROWS_IN_PROMPT], default=str)[:1500]
        except Exception:
            sample_block = ""

    roles_block = ""
    if column_semantic_roles:
        try:
            roles_block = json.dumps(column_semantic_roles, default=str)[:1200]
        except Exception:
            roles_block = ""

    return f"""You are a senior enterprise-data architect classifying ONE dataset from a \
business archive. The archive may come from ANY system of record (SAP, Oracle \
EBS, NetSuite, Microsoft Dynamics, Workday, Salesforce, a data warehouse, or a \
bespoke/in-house schema). Classify ONLY from the evidence below. Do not guess a \
vendor you cannot justify from the columns/values.

FILE NAME: {filename}

AI DESCRIPTION: {desc}

COLUMNS (name, type, sample values):
{col_block}

SAMPLE ROWS (JSON): {sample_block or "(none)"}

PRE-DETECTED SEMANTIC ROLES (JSON): {roles_block or "(none)"}

Return ONLY a JSON object with these fields (no prose, no markdown):
{{
  "source_system": "<the system of record you infer, e.g. 'SAP', 'Oracle EBS', \
'NetSuite', 'Custom/Unknown'. Use 'Unknown' if the evidence does not justify a name.>",
  "erp_module": "<functional area, e.g. 'Sales & Distribution', 'Accounts \
Receivable', 'Accounts Payable', 'General Ledger', 'Inventory', 'HR', or 'Unknown'>",
  "domain_polarity": "<exactly one of: customer | vendor | neutral. 'customer' = \
sell-side / money-in / receivables / sales. 'vendor' = buy-side / money-out / \
payables / procurement. 'neutral' = master data, ledger, inventory, HR, or \
anything not clearly one side>",
  "process_role": "<short snake_case slug for this file's role in a business \
process, e.g. 'sales_order', 'sales_order_line', 'ar_invoice', 'vendor_invoice', \
'payment', 'delivery', 'customer_master', 'gl_journal'. Use 'unknown' if unclear>",
  "grain": "<one short sentence: what does ONE ROW represent?>",
  "confidence": <number 0.0-1.0: how sure you are, given ONLY this evidence>,
  "evidence": ["<short reason citing specific columns/values>", "..."]
}}"""


# ── JSON extraction (robust to fences / stray prose) ──────────────────────────

def _extract_json(raw: str) -> dict[str, Any]:
    if not raw:
        return {}
    text = raw.strip()
    # Strip ```json ... ``` fences if present.
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    # Fallback: grab the first balanced {...} block.
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def _coerce(result: dict[str, Any]) -> ErpClassification:
    """Validate + normalise the raw LLM dict into a typed classification."""
    polarity = str(result.get("domain_polarity") or "neutral").strip().lower()
    if polarity not in _VALID_POLARITIES:
        polarity = "neutral"

    try:
        confidence = float(result.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    evidence_raw = result.get("evidence") or []
    if isinstance(evidence_raw, str):
        evidence = [evidence_raw]
    elif isinstance(evidence_raw, list):
        evidence = [str(e) for e in evidence_raw][:6]
    else:
        evidence = []

    def _clean(value: Any, default: str) -> str:
        s = str(value or "").strip()
        return s or default

    return ErpClassification(
        source_system=_clean(result.get("source_system"), "Unknown"),
        erp_module=_clean(result.get("erp_module"), "Unknown"),
        domain_polarity=polarity,  # type: ignore[arg-type]
        process_role=_clean(result.get("process_role"), "unknown").lower().replace(" ", "_"),
        grain=_clean(result.get("grain"), ""),
        confidence=confidence,
        evidence=evidence,
        source="llm",
    )


class ErpClassifier:
    """Stateless, dependency-light classifier. One LLM call per file."""

    def __init__(self, *, confidence_floor: float | None = None) -> None:
        self._floor = confidence_floor if confidence_floor is not None else _confidence_floor()

    async def classify(
        self,
        *,
        filename: str,
        columns_info: list | None = None,
        sample_rows: list | None = None,
        ai_description: str | None = None,
        column_semantic_roles: dict | None = None,
    ) -> ErpClassification:
        prompt = _build_prompt(
            filename=filename,
            columns_info=columns_info or [],
            sample_rows=sample_rows or [],
            ai_description=ai_description,
            column_semantic_roles=column_semantic_roles,
        )

        try:
            raw, model_version = await asyncio.to_thread(self._call_llm, prompt)
        except Exception as exc:  # never break ingestion
            ingest_logger.warning("erp_classification_llm_error", error=str(exc)[:200], file=filename)
            return ErpClassification.unknown(f"llm_error: {str(exc)[:120]}")

        parsed = _extract_json(raw)
        if not parsed:
            return ErpClassification.unknown("llm_returned_no_json")

        clf = _coerce(parsed)
        clf.model_version = model_version

        # ── Corroborate polarity against pre-detected semantic roles ──────────
        role_polarity, role_hits = _role_polarity_signal(column_semantic_roles)
        if role_polarity != "neutral":
            if clf.domain_polarity == "neutral":
                # The LLM was unsure but the roles point one way — adopt it,
                # modestly, and record why.
                clf.domain_polarity = role_polarity
                clf.evidence.append(f"polarity from semantic roles: {role_hits}")
            elif clf.domain_polarity != role_polarity:
                # Genuine disagreement — keep the LLM's call but penalise
                # confidence and flag it for human review in the enrich loop.
                clf.confidence = round(clf.confidence * 0.6, 4)
                clf.evidence.append(
                    f"polarity disagreement: llm={clf.domain_polarity} vs roles={role_polarity} {role_hits}"
                )

        # ── Degrade below the floor to a safe 'unknown' ───────────────────────
        if clf.confidence < self._floor:
            degraded = ErpClassification.unknown(
                f"below_confidence_floor ({clf.confidence:.2f} < {self._floor:.2f})"
            )
            # Preserve the model's best guess as evidence for the enrich loop,
            # but mark the record itself unreliable so no gate acts on it.
            degraded.evidence.extend(
                [f"llm_guess_system={clf.source_system}", f"llm_guess_polarity={clf.domain_polarity}"]
            )
            degraded.model_version = model_version
            return degraded

        return clf

    @staticmethod
    def _call_llm(prompt: str) -> tuple[str, str]:
        """Blocking Azure OpenAI call. Returns (raw_text, deployment_name).

        Mirrors the ingestion description-call pattern: get_client() →
        chat.completions.create(...). Requests JSON object output where the
        deployment supports it; falls back transparently otherwise.
        """
        from app.core.openai_client import get_client  # local import — matches llm_tasks pattern

        client, deployment = get_client()
        kwargs: dict[str, Any] = {
            "model": deployment,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }
        try:
            resp = client.chat.completions.create(
                response_format={"type": "json_object"}, **kwargs
            )
        except Exception:
            # Older deployments may reject response_format — retry without it.
            resp = client.chat.completions.create(**kwargs)
        raw = resp.choices[0].message.content or ""
        return raw, str(deployment)


def schema_fingerprint(columns_info: list | None) -> str:
    """Stable hash of a file's column shape (names + types, order-independent).

    Two files with the same fingerprint have the same schema, so one's
    classification can be reused for the other — the cache that lets large ERP
    archives (many identical standard tables) skip most LLM calls on re-ingest.
    """
    import hashlib  # local — keep module import surface small

    parts: list[str] = []
    for col in (columns_info or []):
        if isinstance(col, dict):
            name = str(col.get("name") or col.get("column") or "").strip().lower()
            ctype = str(col.get("type") or col.get("dtype") or "").strip().lower()
            if name:
                parts.append(f"{name}:{ctype}")
        elif isinstance(col, str):
            parts.append(col.strip().lower())
    if not parts:
        return ""
    blob = "|".join(sorted(parts))
    return hashlib.sha256(blob.encode()).hexdigest()


async def classify_file(
    *,
    filename: str,
    columns_info: list | None = None,
    sample_rows: list | None = None,
    ai_description: str | None = None,
    column_semantic_roles: dict | None = None,
) -> ErpClassification:
    """Convenience wrapper — one-shot classification with default config."""
    return await ErpClassifier().classify(
        filename=filename,
        columns_info=columns_info,
        sample_rows=sample_rows,
        ai_description=ai_description,
        column_semantic_roles=column_semantic_roles,
    )
