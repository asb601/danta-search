"""Regenerate ACCURATE ``good_for`` for the 18 demo canonical-master tables.

WHY: on this synthetic OEBS dataset every table was templated from the SAME
schema, so the ingestion-time ``good_for`` (generated from that identical schema)
is invoice-flavored for tables that are NOT invoices (AP_CHECKS, AR_CASH_RECEIPTS,
…). For the 18 demo MASTER tables we regenerate ``good_for`` grounded in each
table's TRUE purpose from the GROUND-TRUTH CATALOG (table name + Oracle module +
module description + key columns) and the table's ACTUAL column list — NOT the
templated schema. One gpt-4o-mini call per table (temp 0, json_object).

Mapping chain (verified, 1:1):
    SemanticEntity.is_canonical_master==True (container)
      -> file_id
      -> File.name                       (e.g. "AR_CASH_RECEIPTS_ALL.txt")
      -> OEBS_Table = File.name w/o ext  (e.g. "AR_CASH_RECEIPTS_ALL")
      -> catalog row (OEBS_Table -> Oracle_Module, Module_Description,
                      Key_Columns, Description)
      -> FileMetadata.columns_info       (actual column names/types)
A table that has no catalog row, no FileMetadata, or whose LLM call fails is
SKIPPED and reported (never written with a fabricated good_for).

Run from server/:
    uv run python -m testing.kb_demo_goodfor            # dry-run (DEFAULT, no writes)
    uv run python -m testing.kb_demo_goodfor --apply     # write good_for + commit
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import sys
from dataclasses import dataclass, field

from sqlalchemy import select

# Make sure server/ is on the path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.config import get_settings
from app.core.openai_client import get_client
from app.models.file import File
from app.models.file_metadata import FileMetadata
from app.models.semantic_layer import SemanticEntity

# ── Constants ────────────────────────────────────────────────────────────────
CONTAINER_ID = "9a759559-446f-4751-8167-26a174d05de8"

_DOCS = "/Users/bharath/Desktop/projects/G-CHAT-/docs/superpowers/plans"
CATALOG_CSV = f"{_DOCS}/2026-06-11-demo-table-catalog.csv"

# How many actual columns to show the LLM. The full list grounds the questions in
# REAL column names; this cap is a size guard, not a business knob.
_MAX_COLUMNS = 40
_GOODFOR_MIN = 3
_GOODFOR_MAX = 5


# ── CSV loading ──────────────────────────────────────────────────────────────
def load_catalog() -> dict[str, dict]:
    """OEBS_Table (upper) -> catalog row dict."""
    rows: dict[str, dict] = {}
    with open(CATALOG_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("OEBS_Table") or "").strip().upper()
            if key:
                rows[key] = row
    return rows


def oebs_table_from_filename(name: str) -> str:
    """OEBS_Table key from a DB File.name: strip the file extension, upper-case.

    e.g. 'AR_CASH_RECEIPTS_ALL.txt' -> 'AR_CASH_RECEIPTS_ALL'. The catalog keys
    on the bare OEBS_Table name; File.name == catalog File_Name (with extension)."""
    base = (name or "").strip()
    # Strip the LAST extension only (table names have no dots; the suffix is .xlsx/.csv/.txt/.xls).
    if "." in base:
        base = base.rsplit(".", 1)[0]
    return base.upper()


def column_names(columns_info: list | None, cap: int = _MAX_COLUMNS) -> list[str]:
    """Ordered actual column names from FileMetadata.columns_info (list of dicts)."""
    out: list[str] = []
    for c in columns_info or []:
        if isinstance(c, dict):
            nm = (c.get("name") or "").strip()
        else:
            nm = str(c).strip()
        if nm:
            out.append(nm)
        if len(out) >= cap:
            break
    return out


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class TablePlan:
    file_id: str
    file_name: str                       # DB File.name (with extension)
    oebs_table: str                      # File.name without extension (catalog key)
    label: str | None = None             # SemanticEntity.master_for_entity
    in_catalog: bool = False
    oracle_module: str | None = None
    module_description: str | None = None
    key_columns: str | None = None       # raw catalog Key_Columns cell
    catalog_description: str | None = None
    columns: list[str] = field(default_factory=list)
    old_good_for: list | None = None
    new_good_for: list[str] | None = None
    skip_reason: str | None = None


# ── Prompt + one mini call per table ─────────────────────────────────────────
def _good_for_prompt(tp: TablePlan) -> str:
    """Build the per-table prompt. Grounds the LLM in the table's TRUE identity
    (catalog name + Oracle module + module description + key columns) and its
    ACTUAL column list, and EXPLICITLY warns against schema-templated drift so a
    payments/checks/receipts table does not get invoice-flavored questions."""
    cols = ", ".join(tp.columns) if tp.columns else "(no column metadata)"
    return f"""You are a senior Oracle E-Business Suite (EBS / OEBS) data analyst. For ONE
authoritative master table, write the specific business questions this table can
authoritatively answer.

This is a synthetic dataset where many tables were generated from a SHARED schema
template, so column lists alone can be misleading. Decide what this table is for
from its IDENTITY — the Oracle table name + Oracle module + module description —
NOT from how generic the columns look. Example: a CASH RECEIPTS table answers
questions about CASH RECEIVED / payments / receipt application, NOT about invoices;
a CHECKS table answers payment/disbursement questions, NOT invoice questions; a
TAX (ZX) table answers tax questions; an ON-HAND table answers inventory-balance
questions.

TABLE IDENTITY (the ground truth — trust this over the columns):
  Oracle table   : {tp.oebs_table}
  Oracle module  : {tp.oracle_module} ({tp.module_description})
  catalog purpose: {tp.catalog_description}
  key columns    : {tp.key_columns}

ACTUAL COLUMNS in this table (use REAL names from here in your questions):
  {cols}

Return ONLY JSON of this exact shape:
{{
  "good_for": [
    "<a specific business question this table authoritatively answers>",
    "..."
  ]
}}
Rules:
- Return between {_GOODFOR_MIN} and {_GOODFOR_MAX} questions.
- Each question must be SPECIFIC to THIS table's true purpose (its module + name),
  phrased as a real analyst question, and must reference one or more REAL column
  names from the ACTUAL COLUMNS list above.
- Do NOT invent column names. Do NOT write invoice questions unless this table is
  truly an invoice table. Do NOT restate the schema generically.
- Output JSON only, no prose."""


def _propose_good_for(tp: TablePlan) -> list[str] | None:
    """ONE gpt-4o-mini call (temp 0, json_object) -> a list[str] of 3-5 questions,
    or None on any LLM/JSON error or an unusable shape. Never raises."""
    try:
        client, _ = get_client()
        deployment = get_settings().AZURE_OPENAI_DEPLOYMENT_MINI
        resp = client.chat.completions.create(
            model=deployment,
            messages=[{"role": "user", "content": _good_for_prompt(tp)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_completion_tokens=700,
        )
        raw = (resp.choices[0].message.content or "{}").strip()
        out = json.loads(raw)
    except Exception as exc:  # noqa: BLE001 — skip-and-report, never raise
        tp.skip_reason = f"LLM/JSON error: {str(exc)[:160]}"
        return None

    items = out.get("good_for") if isinstance(out, dict) else None
    if not isinstance(items, list):
        tp.skip_reason = "LLM returned no good_for list"
        return None
    cleaned = [str(q).strip() for q in items if isinstance(q, (str, int, float)) and str(q).strip()]
    if len(cleaned) < _GOODFOR_MIN:
        tp.skip_reason = f"LLM returned only {len(cleaned)} question(s) (<{_GOODFOR_MIN})"
        return None
    return cleaned[:_GOODFOR_MAX]


# ── Build the plan (resolve 18 masters -> catalog + columns) ─────────────────
async def build_plan(db, catalog: dict[str, dict]) -> list[TablePlan]:
    """Resolve the 18 canonical-master tables to a verified per-table plan."""
    ents = (
        await db.execute(
            select(SemanticEntity).where(
                SemanticEntity.container_id == CONTAINER_ID,
                SemanticEntity.is_canonical_master.is_(True),
            )
        )
    ).scalars().all()

    fids = [e.file_id for e in ents]
    files = (
        await db.execute(select(File).where(File.id.in_(fids)))
    ).scalars().all()
    fmap = {f.id: f for f in files}
    metas = (
        await db.execute(select(FileMetadata).where(FileMetadata.file_id.in_(fids)))
    ).scalars().all()
    mmap = {m.file_id: m for m in metas}

    plans: list[TablePlan] = []
    for e in ents:
        f = fmap.get(e.file_id)
        fname = getattr(f, "name", None) or ""
        oebs = oebs_table_from_filename(fname)
        tp = TablePlan(
            file_id=e.file_id,
            file_name=fname,
            oebs_table=oebs,
            label=e.master_for_entity or e.entity_name,
        )
        meta = mmap.get(e.file_id)
        tp.old_good_for = (meta.good_for if meta else None)
        tp.columns = column_names(meta.columns_info if meta else None)

        crow = catalog.get(oebs)
        if not crow:
            tp.skip_reason = f"OEBS_Table {oebs!r} not in ground-truth catalog"
            plans.append(tp)
            continue
        tp.in_catalog = True
        tp.oracle_module = (crow.get("Oracle_Module") or "").strip()
        tp.module_description = (crow.get("Module_Description") or "").strip()
        tp.key_columns = (crow.get("Key_Columns") or "").strip()
        tp.catalog_description = (crow.get("Description") or "").strip()

        if meta is None:
            tp.skip_reason = "no FileMetadata row (cannot read columns / write good_for)"
            plans.append(tp)
            continue

        plans.append(tp)

    # Deterministic order by OEBS_Table for stable reporting.
    plans.sort(key=lambda t: t.oebs_table)
    return plans


async def generate_good_for(plans: list[TablePlan]) -> None:
    """Fill ``new_good_for`` for every catalog-mapped plan via one mini call each.

    Runs in worker threads (the OpenAI client is sync) so the calls overlap; each
    failure is captured on its own TablePlan.skip_reason (never raises)."""
    targets = [tp for tp in plans if tp.in_catalog and tp.skip_reason is None]

    async def _one(tp: TablePlan) -> None:
        tp.new_good_for = await asyncio.to_thread(_propose_good_for, tp)

    await asyncio.gather(*(_one(tp) for tp in targets))


# ── Apply (only verified, only with --apply) ─────────────────────────────────
async def apply_plan(db, plans: list[TablePlan]) -> int:
    """Write new_good_for to FileMetadata.good_for for every successfully
    generated table. Returns the count written."""
    written = 0
    for tp in plans:
        if tp.skip_reason or not tp.new_good_for:
            continue
        meta = (
            await db.execute(
                select(FileMetadata).where(FileMetadata.file_id == tp.file_id)
            )
        ).scalar_one_or_none()
        if meta is None:
            tp.skip_reason = "FileMetadata row vanished before write"
            continue
        meta.good_for = list(tp.new_good_for)
        written += 1
    await db.commit()
    return written


# ── Reporting ────────────────────────────────────────────────────────────────
def _fmt_list(items: list | None) -> str:
    if not items:
        return "(none)"
    return items[0] if isinstance(items, list) else str(items)


def print_report(plans: list[TablePlan]) -> None:
    print("\n" + "=" * 90)
    print(f"GOOD_FOR REGENERATION — {len(plans)} canonical-master tables")
    print("=" * 90)
    for tp in plans:
        head = f"{tp.oebs_table}  [{tp.oracle_module or '?'}]  (file={tp.file_name})"
        print("\n" + head)
        print("-" * len(head))
        if tp.skip_reason:
            print(f"  SKIP: {tp.skip_reason}")
            # Still show the old value so the reviewer sees what stays.
            print(f"  OLD good_for[0]: {_fmt_list(tp.old_good_for)}")
            continue
        print(f"  OLD good_for[0]: {_fmt_list(tp.old_good_for)}")
        for i, q in enumerate(tp.new_good_for or []):
            print(f"  NEW good_for[{i}]: {q}")


def print_summary(plans: list[TablePlan]) -> None:
    ok = [tp for tp in plans if not tp.skip_reason and tp.new_good_for]
    skipped = [tp for tp in plans if tp.skip_reason]
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"  tables (canonical masters): {len(plans)}")
    print(f"  good_for generated        : {len(ok)}")
    print(f"  skipped                   : {len(skipped)}")
    for tp in skipped:
        print(f"    - {tp.oebs_table}: {tp.skip_reason}")


# ── Main ─────────────────────────────────────────────────────────────────────
async def main(apply: bool) -> None:
    catalog = load_catalog()
    async with async_session_ctx() as db:
        plans = await build_plan(db, catalog)
        await generate_good_for(plans)

        print_report(plans)
        print_summary(plans)

        if not apply:
            print("\n[DRY-RUN] No writes performed. Re-run with --apply to commit.\n")
            return

        written = await apply_plan(db, plans)
        print("\n" + "=" * 90)
        print(f"APPLIED — good_for written to FileMetadata for {written} table(s).")
        print("=" * 90 + "\n")


async def verify() -> None:
    """Read-only audit: read the PERSISTED good_for for the 18 masters straight
    from the DB (no LLM) and confirm (a) every master has >=3 questions and
    (b) no NON-invoice table is invoice-flavored. Reproducible within this module."""
    async with async_session_ctx() as db:
        ents = (
            await db.execute(
                select(SemanticEntity).where(
                    SemanticEntity.container_id == CONTAINER_ID,
                    SemanticEntity.is_canonical_master.is_(True),
                )
            )
        ).scalars().all()
        fids = [e.file_id for e in ents]
        fmap = {f.id: f for f in (await db.execute(select(File).where(File.id.in_(fids)))).scalars().all()}
        mmap = {m.file_id: m for m in (await db.execute(select(FileMetadata).where(FileMetadata.file_id.in_(fids)))).scalars().all()}

        ok = 0
        short: list[str] = []
        invoice_flavored: list[str] = []
        for e in ents:
            name = (getattr(fmap.get(e.file_id), "name", "") or "")
            meta = mmap.get(e.file_id)
            gf = (meta.good_for if meta else None) or []
            if isinstance(gf, list) and len(gf) >= _GOODFOR_MIN:
                ok += 1
            else:
                short.append(f"{name} (n={len(gf) if isinstance(gf, list) else 0})")
            is_invoice_table = name.upper().startswith("AP_INVOICES")
            if not is_invoice_table:
                n = sum(1 for q in gf if isinstance(q, str) and "invoice" in q.lower())
                if n:
                    invoice_flavored.append(f"{name} ({n}/{len(gf)} mention 'invoice')")

        print("\n" + "=" * 90)
        print("VERIFY (read-only, from DB)")
        print("=" * 90)
        print(f"  canonical masters           : {len(ents)}")
        print(f"  with valid good_for (>= {_GOODFOR_MIN})   : {ok}")
        print(f"  short/empty                 : {short or '(none)'}")
        print(f"  non-invoice tables mentioning 'invoice': {invoice_flavored or '(none)'}")
        print("=" * 90 + "\n")


def async_session_ctx():
    # Local import keeps the module importable even if the DB layer is heavy.
    from app.core.database import async_session

    return async_session()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="Write good_for to FileMetadata (default is dry-run: no writes).",
    )
    ap.add_argument(
        "--verify", action="store_true",
        help="Read-only: audit the PERSISTED good_for in the DB (no LLM, no writes).",
    )
    args = ap.parse_args()
    if args.verify:
        asyncio.run(verify())
    else:
        asyncio.run(main(apply=args.apply))
