"""ERP-aware join resolution for 7 cross-domain pairs in the OEBS demo container.

A naive first-column / same-name matcher skips these because the genuine Oracle
EBS join is on a DIFFERENTLY-NAMED or non-first key (e.g. OE.LINE_ID =
WSH.SOURCE_LINE_ID), or because there IS no usable shared key (subledger->GL is
via XLA; AP tax/project links live at distribution grain, not on the header).

Each pair is resolved to its DB `files.name` (+ file_id) inside the container by
exact filename match (the DB name carries the source extension). We then read
BOTH tables' ACTUAL `FileMetadata.columns_info` and, using Oracle EBS knowledge,
choose the genuine join key.

NO-FABRICATION RULE (hard): a join is DECLARED only if BOTH chosen columns
ACTUALLY EXIST in their table's columns_info. If the real ERP join key is not
present in both tables, we ABSTAIN (report "no usable join key") rather than
invent one — an honest abstain is correct, not a failure.

A declared row is written as an APPROVED, ACTIVE `SemanticRelationship`
(relationship_type='many_to_one').

Run from server/:
    uv run python -m testing.kb_demo_joins            # dry-run (default)
    uv run python -m testing.kb_demo_joins --apply     # commit verified joins
"""

import argparse
import asyncio
import uuid

from sqlalchemy import select

from app.core.database import async_session
from app.models.file import File
from app.models.file_metadata import FileMetadata

# FileRelationship must be imported so SemanticRelationship's FK to
# file_relationships.id resolves on the SQLAlchemy mapper.
from app.models.file_relationship import FileRelationship  # noqa: F401
from app.models.semantic_layer import SemanticRelationship

CONTAINER_ID = "9a759559-446f-4751-8167-26a174d05de8"


# Each candidate pair. file_a/file_b are the EXACT DB filenames (resolves the
# ambiguous prefixes RCV_TRANSACTIONS* and ZX_LINES*). from_col/to_col are the
# Oracle-EBS join keys we will VERIFY against actual columns_info before any
# declaration. `expect` documents the intended outcome for eyeballing only — the
# actual declare/abstain decision is driven purely by column existence below.
PAIRS = [
    {
        "id": 1,
        "entity_a": "OE_ORDER_LINES_ALL",
        "entity_b": "WSH_DELIVERY_DETAILS",
        "file_a": "OE_ORDER_LINES_ALL.xls",
        "file_b": "WSH_DELIVERY_DETAILS.xls",
        # WSH delivery detail (many) -> OE order line (one), line-grain link.
        "from_entity": "WSH_DELIVERY_DETAILS",
        "from_file": "WSH_DELIVERY_DETAILS.xls",
        "from_col": "SOURCE_LINE_ID",
        "to_entity": "OE_ORDER_LINES_ALL",
        "to_file": "OE_ORDER_LINES_ALL.xls",
        "to_col": "LINE_ID",
        "rationale": "WSH.SOURCE_LINE_ID -> OE_LINES.LINE_ID (delivery detail per order line; SOURCE_CODE='OE')",
        "expect": "declare",
    },
    {
        "id": 2,
        "entity_a": "AR_CASH_RECEIPTS_ALL",
        "entity_b": "GL_JE_LINES",
        "file_a": "AR_CASH_RECEIPTS_ALL.txt",
        "file_b": "GL_JE_LINES.csv",
        # No direct subledger->GL key (linked via XLA in real EBS). The only
        # common column is SET_OF_BOOKS_ID, a books-of-account scoping dimension
        # shared by every finance table — NOT a transaction join key.
        "from_entity": "AR_CASH_RECEIPTS_ALL",
        "from_file": "AR_CASH_RECEIPTS_ALL.txt",
        "from_col": None,
        "to_entity": "GL_JE_LINES",
        "to_file": "GL_JE_LINES.csv",
        "to_col": None,
        "rationale": "AR cash receipts post to GL only via XLA subledger accounting; no shared transaction key (SET_OF_BOOKS_ID is a scope dim, not a join key)",
        "expect": "abstain",
    },
    {
        "id": 3,
        "entity_a": "OE_ORDER_HEADERS_ALL",
        "entity_b": "AR_TRANSACTIONS_ALL",
        "file_a": "OE_ORDER_HEADERS_ALL.xlsx",
        "file_b": "AR_TRANSACTIONS_ALL.csv",
        # Shared CUSTOMER_ID dimension: many AR transactions (many) -> ... ;
        # both sides are many-vs-customer. We orient AR_TRANSACTIONS (txn grain)
        # as the "from" (many) side onto the order header's customer.
        "from_entity": "AR_TRANSACTIONS_ALL",
        "from_file": "AR_TRANSACTIONS_ALL.csv",
        "from_col": "CUSTOMER_ID",
        "to_entity": "OE_ORDER_HEADERS_ALL",
        "to_file": "OE_ORDER_HEADERS_ALL.xlsx",
        "to_col": "CUSTOMER_ID",
        "rationale": "Shared CUSTOMER_ID business key (sold-to customer) links AR transactions and sales orders",
        "expect": "declare",
    },
    {
        "id": 4,
        "entity_a": "CS_SERVICE_REQUESTS",
        "entity_b": "MTL_SYSTEM_ITEMS_B",
        "file_a": "CS_SERVICE_REQUESTS.txt",
        "file_b": "MTL_SYSTEM_ITEMS_B.xls",
        # Service request (many) -> item (one) on INVENTORY_ITEM_ID.
        "from_entity": "CS_SERVICE_REQUESTS",
        "from_file": "CS_SERVICE_REQUESTS.txt",
        "from_col": "INVENTORY_ITEM_ID",
        "to_entity": "MTL_SYSTEM_ITEMS_B",
        "to_file": "MTL_SYSTEM_ITEMS_B.xls",
        "to_col": "INVENTORY_ITEM_ID",
        "rationale": "CS service request references the inventory item via INVENTORY_ITEM_ID",
        "expect": "declare",
    },
    {
        "id": 5,
        "entity_a": "PA_PROJECTS_ALL",
        "entity_b": "AP_INVOICES_ALL",
        "file_a": "PA_PROJECTS_ALL.csv",
        "file_b": "AP_INVOICES_ALL.xlsx",
        # AP_INVOICES header carries no PROJECT_ID; project allocation lives at
        # AP_INVOICE_DISTRIBUTIONS grain in real EBS, absent here.
        "from_entity": "AP_INVOICES_ALL",
        "from_file": "AP_INVOICES_ALL.xlsx",
        "from_col": "PROJECT_ID",
        "to_entity": "PA_PROJECTS_ALL",
        "to_file": "PA_PROJECTS_ALL.csv",
        "to_col": "PROJECT_ID",
        "rationale": "AP invoices link to projects only via distributions (not header); AP_INVOICES_ALL has no PROJECT_ID column",
        "expect": "abstain",
    },
    {
        "id": 6,
        "entity_a": "RCV_TRANSACTIONS",
        "entity_b": "PO_HEADERS_ALL",
        "file_a": "RCV_TRANSACTIONS.xls",
        "file_b": "PO_HEADERS_ALL.xlsx",
        # Receipt (many) -> PO header (one) on PO_HEADER_ID.
        "from_entity": "RCV_TRANSACTIONS",
        "from_file": "RCV_TRANSACTIONS.xls",
        "from_col": "PO_HEADER_ID",
        "to_entity": "PO_HEADERS_ALL",
        "to_file": "PO_HEADERS_ALL.xlsx",
        "to_col": "PO_HEADER_ID",
        "rationale": "RCV receipt references its purchase order via PO_HEADER_ID",
        "expect": "declare",
    },
    {
        "id": 7,
        "entity_a": "AP_INVOICES_ALL",
        "entity_b": "ZX_LINES",
        "file_a": "AP_INVOICES_ALL.xlsx",
        "file_b": "ZX_LINES.csv",
        # ZX tax lines reference the source doc via TRX_ID/APPLICATION_ID/
        # ENTITY_CODE in real EBS, but ZX_LINES here has none of TRX_ID/
        # INVOICE_ID, and AP_INVOICES has no tax-link column. No shared key.
        "from_entity": "ZX_LINES",
        "from_file": "ZX_LINES.csv",
        "from_col": None,
        "to_entity": "AP_INVOICES_ALL",
        "to_file": "AP_INVOICES_ALL.xlsx",
        "to_col": None,
        "rationale": "ZX tax lines link to AP invoices via TRX_ID/APPLICATION_ID (absent in ZX_LINES here) — no shared key column in both",
        "expect": "abstain",
    },
]


async def _resolve_file_columns(db):
    """Return {filename: (file_id, set(column_names))} for every distinct file in PAIRS."""
    wanted = set()
    for p in PAIRS:
        wanted.add(p["file_a"])
        wanted.add(p["file_b"])

    out = {}
    for fname in sorted(wanted):
        fid = (
            await db.execute(
                select(File.id).where(
                    File.container_id == CONTAINER_ID, File.name == fname
                )
            )
        ).scalar_one_or_none()
        if fid is None:
            out[fname] = (None, set())
            continue
        cols_info = (
            await db.execute(
                select(FileMetadata.columns_info).where(FileMetadata.file_id == fid)
            )
        ).scalar_one_or_none()
        col_names = {c.get("name") for c in (cols_info or []) if c.get("name")}
        out[fname] = (fid, col_names)
    return out


async def _existing_relationship(db, file_a_id, file_b_id, from_col, to_col):
    """Detect an already-present SemanticRelationship for this pair+columns (idempotency)."""
    rows = (
        await db.execute(
            select(SemanticRelationship).where(
                SemanticRelationship.container_id == CONTAINER_ID,
                SemanticRelationship.file_a_id.in_([file_a_id, file_b_id]),
                SemanticRelationship.file_b_id.in_([file_a_id, file_b_id]),
            )
        )
    ).scalars().all()
    for r in rows:
        cols = {r.from_column, r.to_column}
        if cols == {from_col, to_col}:
            return r
    return None


async def main(apply: bool):
    declared, abstained = 0, 0
    print(f"\n{'=' * 78}")
    print(f"ERP-aware join resolution — container {CONTAINER_ID}")
    print(f"mode: {'APPLY (writing approved SemanticRelationship rows)' if apply else 'DRY-RUN'}")
    print(f"{'=' * 78}\n")

    async with async_session() as db:
        files = await _resolve_file_columns(db)

        # Per-file resolution summary (file_id + column count) for the audit trail.
        print("Resolved files (filename -> file_id, n_columns):")
        for fname, (fid, cols) in files.items():
            print(f"  {fname:<32} {fid}  ({len(cols)} cols)")
        print()

        to_write = []
        for p in PAIRS:
            fa_id, fa_cols = files[p["from_file"]]
            fb_id, fb_cols = files[p["to_file"]]
            from_col, to_col = p["from_col"], p["to_col"]

            print(f"--- Pair {p['id']}: {p['entity_a']} <-> {p['entity_b']} ---")
            print(f"    rationale: {p['rationale']}")

            # Resolution guard: both files must exist.
            if fa_id is None or fb_id is None:
                missing = p["from_file"] if fa_id is None else p["to_file"]
                print(f"    RESULT: ABSTAIN — file not found in container: {missing}\n")
                abstained += 1
                continue

            # ERP analysis chose no key (genuine no-join pair).
            if from_col is None or to_col is None:
                print(f"    chosen key: (none)")
                print(f"    RESULT: ABSTAIN — no usable join key\n")
                abstained += 1
                continue

            # NO-FABRICATION exist-check: BOTH columns must be present.
            from_ok = from_col in fa_cols
            to_ok = to_col in fb_cols
            print(
                f"    chosen key: {p['from_entity']}.{from_col} = {p['to_entity']}.{to_col}"
            )
            print(
                f"    exists? {p['from_entity']}.{from_col}={from_ok}  "
                f"{p['to_entity']}.{to_col}={to_ok}"
            )

            if not (from_ok and to_ok):
                print(f"    RESULT: ABSTAIN — chosen column missing in a table (no fabrication)\n")
                abstained += 1
                continue

            existing = await _existing_relationship(db, fa_id, fb_id, from_col, to_col)
            if existing is not None:
                print(
                    f"    RESULT: DECLARE (already present, id={existing.id}, "
                    f"status={existing.approval_status}/{existing.status}) — skip write\n"
                )
                declared += 1
                continue

            print(f"    RESULT: DECLARE many_to_one  ({'will write' if apply else 'dry-run'})\n")
            declared += 1
            to_write.append(
                SemanticRelationship(
                    id=str(uuid.uuid4()),
                    container_id=CONTAINER_ID,
                    source_relationship_id=None,
                    file_a_id=fa_id,
                    file_b_id=fb_id,
                    from_entity=p["from_entity"],
                    to_entity=p["to_entity"],
                    from_column=from_col,
                    to_column=to_col,
                    relationship_type="many_to_one",
                    join_rule={
                        "from": {"file": p["from_file"], "column": from_col},
                        "to": {"file": p["to_file"], "column": to_col},
                        "source": "erp_knowledge",
                        "rationale": p["rationale"],
                    },
                    approval_status="approved",
                    risk_reason=None,
                    confidence_score=1.0,
                    status="active",
                )
            )

        if apply and to_write:
            db.add_all(to_write)
            await db.commit()
            print(f"COMMITTED {len(to_write)} new SemanticRelationship row(s).")
        elif apply:
            print("Nothing new to write (all declared joins already present).")

    print(f"\n{'=' * 78}")
    print(f"FINAL: declared={declared}  abstained={abstained}  (total {declared + abstained})")
    print(f"{'=' * 78}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--apply",
        action="store_true",
        help="Commit verified joins (default is dry-run).",
    )
    args = ap.parse_args()
    asyncio.run(main(apply=args.apply))
