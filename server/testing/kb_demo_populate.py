"""
Populate the KB map for the demo from the GROUND-TRUTH CATALOG (no FastAPI).

Goal: for the 10 most complex / cross-domain prompts, write the PROVEN facts the
query side reads — canonical masters (P1), value/declared joins (P11), and ledger
polarity (P3) — with STRICT no-mismatch verification. Every change is verified
through an exact 1:1 mapping chain before it is applied; anything ambiguous is
SKIPPED and reported. Better to skip a table than mis-map it.

Mapping chain (each link verified, 1:1):
    prompt Primary Table (an OEBS_Table, no extension)
      -> catalog row (OEBS_Table -> File_Name)
      -> DB files.name (== File_Name, scoped to the container)
      -> file_id
A table that maps to 0 or >1 DB files is SKIPPED (no mismatch).

Run from server/:
    uv run python -m testing.kb_demo_populate            # dry-run (DEFAULT, no writes)
    uv run python -m testing.kb_demo_populate --apply     # verified writes + commit
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import os
import re
import sys
import uuid
from dataclasses import dataclass, field

from sqlalchemy import and_, or_, select

# Make sure server/ is on the path when run directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.core.database import async_session
from app.models.erp_classification import ErpClassification
from app.models.file import File
# Imported so the mapper can resolve the SemanticRelationship FK
# (source_relationship_id -> file_relationships.id) at flush time.
from app.models.file_relationship import FileRelationship  # noqa: F401
from app.models.semantic_layer import SemanticEntity, SemanticRelationship

# ── Constants ────────────────────────────────────────────────────────────────
CONTAINER_ID = "9a759559-446f-4751-8167-26a174d05de8"

_DOCS = "/Users/bharath/Desktop/projects/G-CHAT-/docs/superpowers/plans"
CATALOG_CSV = f"{_DOCS}/2026-06-11-demo-table-catalog.csv"
PROMPTS_CSV = f"{_DOCS}/2026-06-11-demo-prompts.csv"

# Label derivation: strip a leading Oracle-module prefix and a trailing
# table-suffix, lowercase. e.g. AP_INVOICES_ALL -> invoices,
# RA_CUSTOMER_TRX_ALL -> customer_trx, GL_BALANCES -> balances.
_MODULE_PREFIXES = (
    "AP_", "AR_", "GL_", "PO_", "RA_", "MTL_", "PER_", "PAY_",
    "OE_", "WSH_", "FA_", "PA_", "HZ_",
)
_SUFFIXES = ("_ALL", "_F", "_B", "_TL", "_V")

# Polarity by Oracle module (P3). human_override is always reliable.
_VENDOR_MODULES = {"AP", "PO", "PON", "RCV", "IBY"}
_CUSTOMER_MODULES = {"AR", "OE", "ONT", "WSH", "HZ", "ASO", "QP", "OKC"}

# Split a "Primary Table(s)" cell into individual OEBS_Table names.
_TABLE_SPLIT = re.compile(r"[,&+]")


def derive_label(oebs_table: str) -> str:
    """Short concept token from an OEBS_Table name (strip module prefix +
    trailing suffix, lowercase). Deterministic + dataset-agnostic."""
    name = (oebs_table or "").strip().upper()
    for pre in _MODULE_PREFIXES:
        if name.startswith(pre):
            name = name[len(pre):]
            break
    # Strip ONE trailing suffix (longest match first so _ALL beats _L etc.).
    for suf in sorted(_SUFFIXES, key=len, reverse=True):
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    return name.lower()


def split_primary_tables(cell: str) -> list[str]:
    """OEBS_Table names from a 'Primary Table(s)' cell, in order, de-duped."""
    out: list[str] = []
    seen: set[str] = set()
    for part in _TABLE_SPLIT.split(cell or ""):
        t = part.strip()
        if not t:
            continue
        # Defensive: drop any extension if one ever slips in.
        t = t.split(".")[0].strip()
        key = t.upper()
        if t and key not in seen:
            seen.add(key)
            out.append(t)
    return out


def parse_key_columns(cell: str) -> list[str]:
    """Catalog Key_Columns cell -> ordered list of upper-cased column names."""
    return [c.strip().upper() for c in (cell or "").split(",") if c.strip()]


# ── Data structures ──────────────────────────────────────────────────────────
@dataclass
class TablePlan:
    oebs_table: str
    in_catalog: bool = False
    file_name: str | None = None      # catalog File_Name (with extension)
    oracle_module: str | None = None
    key_columns: list[str] = field(default_factory=list)
    file_id: str | None = None        # resolved DB file id (1:1) or None
    db_match_count: int = 0           # how many DB files matched the name
    label: str | None = None
    polarity: str | None = None       # customer | vendor | None
    skip_reason: str | None = None    # set when this table cannot be used


@dataclass
class JoinPlan:
    prompt_id: str
    table_a: str
    table_b: str
    shared_key: str | None = None
    action: str | None = None         # "approve" | "insert" | "skip"
    skip_reason: str | None = None


# ── CSV loading ──────────────────────────────────────────────────────────────
def load_catalog() -> dict[str, dict]:
    """OEBS_Table -> catalog row dict."""
    rows: dict[str, dict] = {}
    with open(CATALOG_CSV, newline="") as f:
        for row in csv.DictReader(f):
            key = (row.get("OEBS_Table") or "").strip().upper()
            if key:
                rows[key] = row
    return rows


def select_prompts() -> list[dict]:
    """The 10 most complex / cross-domain prompts.

    Prefer Domain == 'Cross-Domain'; fill to 10 with Complexity == 'Complex'.
    Deterministic order: by numeric ID. Cross-Domain prompts come first, then
    Complex ones (excluding any already taken), capped at 10.
    """
    with open(PROMPTS_CSV, newline="") as f:
        all_rows = list(csv.DictReader(f))

    def pid(r: dict) -> int:
        try:
            return int((r.get("ID") or "0").strip())
        except ValueError:
            return 0

    all_rows.sort(key=pid)
    cross = [r for r in all_rows if (r.get("Domain") or "").strip() == "Cross-Domain"]
    complex_rows = [
        r for r in all_rows
        if (r.get("Complexity") or "").strip() == "Complex"
        and (r.get("Domain") or "").strip() != "Cross-Domain"
    ]
    selected = cross[:10]
    for r in complex_rows:
        if len(selected) >= 10:
            break
        selected.append(r)
    selected.sort(key=pid)
    return selected[:10]


# ── DB resolution ────────────────────────────────────────────────────────────
async def resolve_file_id(db, file_name: str) -> tuple[str | None, int]:
    """Resolve a catalog File_Name to exactly one DB file id in the container.

    Returns (file_id_or_None, match_count). Only a count of exactly 1 yields an
    id; 0 or >1 is ambiguous -> caller SKIPs (no mismatch)."""
    ids = (
        await db.execute(
            select(File.id).where(
                File.container_id == CONTAINER_ID,
                File.name == file_name,
            )
        )
    ).scalars().all()
    if len(ids) == 1:
        return ids[0], 1
    return None, len(ids)


# ── Build the verified plan ──────────────────────────────────────────────────
async def build_plan(db, prompts: list[dict], catalog: dict[str, dict]):
    """Produce (table_plans, join_plans) — fully verified, nothing applied."""
    # 1) Collect distinct Primary Tables across the selected prompts.
    table_plans: dict[str, TablePlan] = {}
    for p in prompts:
        for t in split_primary_tables(p.get("Primary Table(s)", "")):
            key = t.upper()
            if key not in table_plans:
                table_plans[key] = TablePlan(oebs_table=t)

    # 2) Verify each table: catalog -> File_Name -> DB file_id (1:1).
    for key, tp in table_plans.items():
        crow = catalog.get(key)
        if not crow:
            tp.skip_reason = "not in ground-truth catalog"
            continue
        tp.in_catalog = True
        tp.file_name = (crow.get("File_Name") or "").strip()
        tp.oracle_module = (crow.get("Oracle_Module") or "").strip().upper()
        tp.key_columns = parse_key_columns(crow.get("Key_Columns", ""))
        tp.label = derive_label(tp.oebs_table)

        if not tp.file_name:
            tp.skip_reason = "catalog row has no File_Name"
            continue
        fid, count = await resolve_file_id(db, tp.file_name)
        tp.db_match_count = count
        if fid is None:
            tp.skip_reason = (
                "no DB file matches name" if count == 0
                else f"{count} DB files match name (ambiguous)"
            )
            continue
        tp.file_id = fid

        # Polarity (only for mapped tables; modules outside both maps stay None).
        mod = tp.oracle_module or ""
        if mod in _VENDOR_MODULES:
            tp.polarity = "vendor"
        elif mod in _CUSTOMER_MODULES:
            tp.polarity = "customer"

    # 3) Joins: for each multi-table prompt, each adjacent table pair must share
    #    a key present in BOTH tables' catalog Key_Columns (else skip — no fab).
    join_plans: list[JoinPlan] = []
    for p in prompts:
        pid = (p.get("ID") or "").strip()
        tables = split_primary_tables(p.get("Primary Table(s)", ""))
        if len(tables) < 2:
            continue
        for a_name, b_name in zip(tables, tables[1:]):
            jp = JoinPlan(prompt_id=pid, table_a=a_name, table_b=b_name)
            tpa = table_plans.get(a_name.upper())
            tpb = table_plans.get(b_name.upper())

            if not tpa or not tpa.file_id:
                jp.action = "skip"
                jp.skip_reason = f"table {a_name} not mapped 1:1"
                join_plans.append(jp)
                continue
            if not tpb or not tpb.file_id:
                jp.action = "skip"
                jp.skip_reason = f"table {b_name} not mapped 1:1"
                join_plans.append(jp)
                continue

            shared = [k for k in tpa.key_columns if k in set(tpb.key_columns)]
            if not shared:
                jp.action = "skip"
                jp.skip_reason = "no shared key in BOTH catalog Key_Columns"
                join_plans.append(jp)
                continue

            # Deterministic pick: first shared key in table A's catalog order.
            jp.shared_key = shared[0]
            # Decide approve-vs-insert by probing existing candidates.
            existing = await _find_candidate_relationship(
                db, tpa.file_id, tpb.file_id, jp.shared_key
            )
            jp.action = "approve" if existing is not None else "insert"
            join_plans.append(jp)

    return table_plans, join_plans


async def _find_candidate_relationship(
    db, fid_a: str, fid_b: str, shared_key: str
) -> SemanticRelationship | None:
    """Find an existing SemanticRelationship between two files on the shared key.

    Pair is matched in EITHER (file_a, file_b) order. We prefer the clean
    exact-key candidate (from_column == to_column == shared_key) over an
    audit-ish variant (e.g. BILL_TO_CUSTOMER_ID -> CUSTOMER_ID), and never
    approve a row whose from_column is not the shared key."""
    rows = (
        await db.execute(
            select(SemanticRelationship).where(
                SemanticRelationship.container_id == CONTAINER_ID,
                or_(
                    and_(
                        SemanticRelationship.file_a_id == fid_a,
                        SemanticRelationship.file_b_id == fid_b,
                    ),
                    and_(
                        SemanticRelationship.file_a_id == fid_b,
                        SemanticRelationship.file_b_id == fid_a,
                    ),
                ),
            )
        )
    ).scalars().all()
    key = shared_key.upper()
    exact = [
        r for r in rows
        if (r.from_column or "").upper() == key and (r.to_column or "").upper() == key
    ]
    if exact:
        return exact[0]
    on_key = [r for r in rows if (r.from_column or "").upper() == key]
    return on_key[0] if on_key else None


# ── Apply (only verified, only with --apply) ─────────────────────────────────
async def apply_plan(db, table_plans, join_plans):
    counts = {
        "masters_set": 0,
        "joins_approved": 0,
        "joins_inserted": 0,
        "polarity_set": 0,
        "skipped": 0,
    }
    mapped = [tp for tp in table_plans.values() if tp.file_id]
    counts["skipped"] = sum(1 for tp in table_plans.values() if tp.skip_reason)

    # P1 — canonical masters. One master per label group.
    for tp in mapped:
        label = tp.label
        # Demote every OTHER entity currently flagged for this label group.
        others = (
            await db.execute(
                select(SemanticEntity).where(
                    SemanticEntity.container_id == CONTAINER_ID,
                    SemanticEntity.master_for_entity == label,
                    SemanticEntity.file_id != tp.file_id,
                )
            )
        ).scalars().all()
        for o in others:
            o.is_canonical_master = False
            o.master_for_entity = None

        ent = (
            await db.execute(
                select(SemanticEntity).where(
                    SemanticEntity.container_id == CONTAINER_ID,
                    SemanticEntity.file_id == tp.file_id,
                )
            )
        ).scalar_one_or_none()
        if ent is None:
            # No entity row for this file — cannot set a master. Report + skip.
            tp.skip_reason = "no SemanticEntity row for mapped file (master not set)"
            counts["skipped"] += 1
            continue
        ent.is_canonical_master = True
        ent.master_for_entity = label
        ent.entity_name = label
        counts["masters_set"] += 1

    # P3 — polarity (human_override, confidence 1.0).
    for tp in mapped:
        if tp.polarity is None:
            continue
        clf = (
            await db.execute(
                select(ErpClassification).where(
                    ErpClassification.container_id == CONTAINER_ID,
                    ErpClassification.file_id == tp.file_id,
                )
            )
        ).scalar_one_or_none()
        if clf is None:
            clf = ErpClassification(
                id=str(uuid.uuid4()),
                container_id=CONTAINER_ID,
                file_id=tp.file_id,
            )
            db.add(clf)
        clf.domain_polarity = tp.polarity
        clf.source = "human_override"
        clf.confidence = 1.0
        counts["polarity_set"] += 1

    # P11 — joins. Approve an existing candidate, else insert an approved edge.
    for jp in join_plans:
        if jp.action not in ("approve", "insert"):
            continue
        tpa = table_plans[jp.table_a.upper()]
        tpb = table_plans[jp.table_b.upper()]
        if jp.action == "approve":
            rel = await _find_candidate_relationship(
                db, tpa.file_id, tpb.file_id, jp.shared_key
            )
            if rel is None:
                # Lost the row between plan and apply — fall through to insert.
                jp.action = "insert"
            else:
                rel.approval_status = "approved"
                rel.status = "active"
                counts["joins_approved"] += 1
        if jp.action == "insert":
            db.add(
                SemanticRelationship(
                    id=str(uuid.uuid4()),
                    container_id=CONTAINER_ID,
                    file_a_id=tpa.file_id,
                    file_b_id=tpb.file_id,
                    from_entity=tpa.label,
                    to_entity=tpb.label,
                    from_column=jp.shared_key,
                    to_column=jp.shared_key,
                    relationship_type="many_to_one",
                    approval_status="approved",
                    status="active",
                    confidence_score=1.0,
                )
            )
            counts["joins_inserted"] += 1

    await db.commit()
    return counts


# ── Reporting ────────────────────────────────────────────────────────────────
def print_prompt_selection(prompts: list[dict]) -> None:
    print("\n" + "=" * 78)
    print(f"SELECTED {len(prompts)} PROMPTS (most complex / cross-domain)")
    print("=" * 78)
    for p in prompts:
        print(
            f"  #{(p.get('ID') or '').strip():>2} [{(p.get('Domain') or '').strip()}"
            f" / {(p.get('Complexity') or '').strip()}]"
            f" tables={split_primary_tables(p.get('Primary Table(s)', ''))}"
        )
        print(f"       \"{(p.get('NLP Prompt (what the user types)') or '').strip()}\"")


def print_table_report(table_plans: dict[str, TablePlan]) -> None:
    print("\n" + "-" * 78)
    print("PER PRIMARY TABLE  (mapped 1:1? / label / polarity)")
    print("-" * 78)
    print(f"  {'OEBS_TABLE':32s} {'MAPPED':7s} {'LABEL':16s} {'POLARITY':9s}")
    for key in sorted(table_plans):
        tp = table_plans[key]
        mapped = "YES" if tp.file_id else "NO"
        label = tp.label or "-"
        pol = tp.polarity or "-"
        print(f"  {tp.oebs_table:32s} {mapped:7s} {label:16s} {pol:9s}")


def print_join_report(join_plans: list[JoinPlan]) -> None:
    print("\n" + "-" * 78)
    print("PER JOIN  (pair / shared-key / approve-or-insert)")
    print("-" * 78)
    if not join_plans:
        print("  (no multi-table prompts among the selection)")
        return
    for jp in join_plans:
        if jp.action == "skip":
            print(
                f"  #{jp.prompt_id:>2} {jp.table_a} <-> {jp.table_b}"
                f"  -> SKIP ({jp.skip_reason})"
            )
        else:
            print(
                f"  #{jp.prompt_id:>2} {jp.table_a} <-> {jp.table_b}"
                f"  key={jp.shared_key}  -> {jp.action.upper()}"
            )


def print_skipped(table_plans: dict[str, TablePlan], join_plans: list[JoinPlan]) -> None:
    print("\n" + "-" * 78)
    print("SKIPPED  (the no-mismatch evidence)")
    print("-" * 78)
    any_skip = False
    for key in sorted(table_plans):
        tp = table_plans[key]
        if tp.skip_reason:
            any_skip = True
            print(f"  TABLE {tp.oebs_table}: {tp.skip_reason}")
    for jp in join_plans:
        if jp.action == "skip":
            any_skip = True
            print(f"  JOIN  #{jp.prompt_id} {jp.table_a}<->{jp.table_b}: {jp.skip_reason}")
    if not any_skip:
        print("  (nothing skipped)")


def print_dryrun_summary(table_plans, join_plans) -> None:
    mapped = sum(1 for tp in table_plans.values() if tp.file_id)
    skipped_t = sum(1 for tp in table_plans.values() if tp.skip_reason)
    pol = sum(1 for tp in table_plans.values() if tp.file_id and tp.polarity)
    approve = sum(1 for jp in join_plans if jp.action == "approve")
    insert = sum(1 for jp in join_plans if jp.action == "insert")
    skip_j = sum(1 for jp in join_plans if jp.action == "skip")
    print("\n" + "=" * 78)
    print("DRY-RUN SUMMARY")
    print("=" * 78)
    print(f"  tables: {len(table_plans)}  mapped 1:1: {mapped}  skipped: {skipped_t}")
    print(f"  masters would be set: {mapped}   polarity would be set: {pol}")
    print(f"  joins: approve={approve}  insert={insert}  skip={skip_j}")


# ── Main ─────────────────────────────────────────────────────────────────────
async def main(apply: bool) -> None:
    catalog = load_catalog()
    prompts = select_prompts()
    print_prompt_selection(prompts)

    async with async_session() as db:
        table_plans, join_plans = await build_plan(db, prompts, catalog)

        print_table_report(table_plans)
        print_join_report(join_plans)
        print_skipped(table_plans, join_plans)
        print_dryrun_summary(table_plans, join_plans)

        if not apply:
            print("\n[DRY-RUN] No writes performed. Re-run with --apply to commit.\n")
            return

        counts = await apply_plan(db, table_plans, join_plans)
        print("\n" + "=" * 78)
        print("APPLIED — FINAL COUNTS")
        print("=" * 78)
        for k, v in counts.items():
            print(f"  {k:16s}: {v}")
        print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--apply", action="store_true",
        help="Commit verified writes (default is dry-run: no writes).",
    )
    args = ap.parse_args()
    asyncio.run(main(apply=args.apply))
