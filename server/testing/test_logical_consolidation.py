"""TDD spec for the query-runtime foundation fixes.

Pure-function tests (no DB / no Azure) covering:
  1. Logical-table consolidation — many monthly/format partitions of one table
     family resolve to ONE logical table whose execution spans every partition.
  2. Dialect-safe canonicalization — vendor SQL idioms (DATEDIFF, GROUP_CONCAT)
     transpile to the executor dialect instead of raising a hard parse error.
  3. Typed errors — parse failures are NOT reported as authorization errors.

Run: cd server && uv run pytest testing/test_logical_consolidation.py -q
"""
from __future__ import annotations

import re

from app.services.file_identity import (
    build_file_identity_map,
    logical_table_key,
)
from app.services.logical_sql import (
    canonicalize_logical_sql,
    SQLCanonicalizationError,
    SQLParseError,
    SQLAuthorizationError,
)


def _catalog():
    """A PROC_PurchaseOrders family spread across months AND formats,
    plus one unrelated single-file table."""
    entries = [
        ("f-2023-01", "Procurement/aaa11111_PROC_PurchaseOrders_2023_01_pipe.txt"),
        ("f-2023-02", "Procurement/bbb22222_PROC_PurchaseOrders_2023_02.csv"),
        ("f-2023-06", "Procurement/ccc33333_PROC_PurchaseOrders_2023_06_tab.txt"),
        ("f-2024-04", "Procurement/ddd44444_PROC_PurchaseOrders_2024_04.csv"),
        ("f-ar-only", "Finance/eee55555_AR_DSO_Analysis_2023_01.csv"),
    ]
    catalog = [{"file_id": fid, "blob_path": bp} for fid, bp in entries]
    parquet = {bp: bp.rsplit(".", 1)[0] + ".parquet" for _, bp in entries}
    return catalog, parquet


# ── 1. Logical-table key derivation ─────────────────────────────────────────
def test_logical_table_key_strips_period_and_format():
    assert logical_table_key("x/aaa11111_PROC_PurchaseOrders_2023_06_tab.txt") == "PROC_PURCHASEORDERS"
    assert logical_table_key("x/bbb22222_PROC_PurchaseOrders_2023_02.csv") == "PROC_PURCHASEORDERS"
    assert logical_table_key("x/ccc33333_PROC_PurchaseOrders_2024_04_pipe.txt") == "PROC_PURCHASEORDERS"
    # unrelated family is distinct
    assert logical_table_key("x/eee55555_AR_DSO_Analysis_2023_01.csv") == "AR_DSO_ANALYSIS"


# ── 2. Consolidation: one logical table, all partitions authorized ───────────
def test_partitions_consolidate_to_one_logical_table():
    catalog, parquet = _catalog()
    m = build_file_identity_map(catalog, parquet, "cont1")

    ident = m.resolve_table("PROC_PURCHASEORDERS")
    # all 4 PROC partitions belong to the one logical table
    assert set(ident.member_file_ids) == {"f-2023-01", "f-2023-02", "f-2023-06", "f-2024-04"}
    assert len(ident.partition_uris) == 4
    # every partition file id is authorized through the map
    allowed = m.allowed_file_ids()
    assert {"f-2023-01", "f-2023-02", "f-2023-06", "f-2024-04", "f-ar-only"} <= allowed


# ── 3. Canonicalized SQL scans every partition ───────────────────────────────
def test_query_over_logical_table_expands_to_all_partitions():
    catalog, parquet = _catalog()
    m = build_file_identity_map(catalog, parquet, "cont1")
    allowed = m.allowed_file_ids()

    out = canonicalize_logical_sql(
        "SELECT Purchasing_Org, SUM(Net_Value) AS spend FROM PROC_PURCHASEORDERS GROUP BY Purchasing_Org",
        m,
        allowed_file_ids=allowed,
    )
    # the executable SQL must reference a read_parquet call per partition,
    # so the executor registers all 4 (the regex matches single-path calls)
    reads = re.findall(r"read_parquet\('([^']+)'\)", out.executable_sql, re.IGNORECASE)
    assert len(reads) == 4
    # all partition file ids are recorded as referenced (authorization breadth)
    assert set(out.referenced_file_ids) == {"f-2023-01", "f-2023-02", "f-2023-06", "f-2024-04"}


# ── 4. Dialect safety: DATEDIFF / GROUP_CONCAT must not hard-crash ───────────
def test_vendor_dialect_idioms_transpile_not_raise():
    catalog, parquet = _catalog()
    m = build_file_identity_map(catalog, parquet, "cont1")
    allowed = m.allowed_file_ids()

    sql = (
        "SELECT Vendor_ID, GROUP_CONCAT(GL_Account) AS gls, "
        "AVG(DATEDIFF(CURRENT_DATE, GR_Date)) AS aging "
        "FROM PROC_PURCHASEORDERS GROUP BY Vendor_ID"
    )
    out = canonicalize_logical_sql(sql, m, allowed_file_ids=allowed)
    assert out.executable_sql  # produced executable SQL rather than raising
    # GROUP_CONCAT should be normalized to duckdb's string_agg
    assert "group_concat" not in out.executable_sql.lower()


# ── 5. Typed errors: parse failure ≠ authorization error ─────────────────────
def test_unparseable_sql_raises_parse_error_not_auth():
    catalog, parquet = _catalog()
    m = build_file_identity_map(catalog, parquet, "cont1")
    raised = None
    try:
        canonicalize_logical_sql("SELECT FROM WHERE GROUP", m, allowed_file_ids=m.allowed_file_ids())
    except SQLCanonicalizationError as exc:
        raised = exc
    assert isinstance(raised, SQLParseError)
    assert not isinstance(raised, SQLAuthorizationError)


# ── 6. SECURITY: partial authorization scans ONLY authorized partitions ──────
def test_partial_authorization_scans_only_authorized_partitions():
    catalog, parquet = _catalog()
    m = build_file_identity_map(catalog, parquet, "cont1")
    # Authorize only two of the four PROC partitions.
    allowed = {"f-2023-01", "f-2023-06"}
    out = canonicalize_logical_sql(
        "SELECT * FROM PROC_PURCHASEORDERS", m, allowed_file_ids=allowed,
    )
    reads = re.findall(r"read_parquet\('([^']+)'\)", out.executable_sql, re.IGNORECASE)
    # exactly the two authorized partitions are scanned — no leak of the other two
    assert len(reads) == 2
    assert set(out.referenced_file_ids) == {"f-2023-01", "f-2023-06"}
    assert all("2023_02" not in u and "2024_04" not in u for u in reads)


# ── 7. Schema gate: divergent-schema partitions do NOT silently merge ────────
def test_schema_incompatible_partitions_are_split():
    # Same logical-table name, but one month has a different column set.
    entries = [
        ("g1", "P/aaaa1111_ORD_2023_01.csv", ["a", "b", "c"]),
        ("g2", "P/bbbb2222_ORD_2023_02.csv", ["a", "b", "c"]),
        ("g3", "P/cccc3333_ORD_2023_03.csv", ["x", "y"]),  # incompatible schema
    ]
    catalog = [
        {"file_id": fid, "blob_path": bp, "column_names": cols}
        for fid, bp, cols in entries
    ]
    parquet = {bp: bp.rsplit(".", 1)[0] + ".parquet" for _, bp, _ in entries}
    m = build_file_identity_map(catalog, parquet, "c")
    ord_ident = m.resolve_table("ORD")
    # majority schema (2 files) keeps the clean name; the odd one is split off
    assert set(ord_ident.member_file_ids) == {"g1", "g2"}
    assert "g3" in m.allowed_file_ids()  # still authorized, just under a split name


# ── 8. Q2: duplicate-format partitions of the same period are deduped ────────
def test_duplicate_format_period_is_deduped():
    entries = [
        ("p1", "P/aaaa1111_ORD_2023_01_pipe.txt", "2023-01-01", "2023-01-31"),
        ("p2", "P/bbbb2222_ORD_2023_01.csv", "2023-01-01", "2023-01-31"),  # SAME period, other format
        ("p3", "P/cccc3333_ORD_2023_02.csv", "2023-02-01", "2023-02-28"),
    ]
    catalog = [
        {"file_id": fid, "blob_path": bp, "date_range_start": s, "date_range_end": e}
        for fid, bp, s, e in entries
    ]
    parquet = {bp: bp.rsplit(".", 1)[0] + ".parquet" for _, bp, _, _ in entries}
    m = build_file_identity_map(catalog, parquet, "c")
    ident = m.resolve_table("ORD")
    # only ONE partition kept for 2023_01 (no double-count) + 2023_02 => 2 total
    assert ident.partition_count == 2
    assert len(ident.partition_uris) == 2


# ── 9. #4: coverage window aggregates across all partitions ──────────────────
def test_coverage_window_aggregates_across_partitions():
    entries = [
        ("c1", "P/aaaa1111_ORD_2022_01.csv", "2022-01-01", "2022-01-31"),
        ("c2", "P/bbbb2222_ORD_2023_06.csv", "2023-06-01", "2023-06-30"),
        ("c3", "P/cccc3333_ORD_2024_12.csv", "2024-12-01", "2024-12-31"),
    ]
    catalog = [
        {"file_id": fid, "blob_path": bp, "date_range_start": s, "date_range_end": e}
        for fid, bp, s, e in entries
    ]
    parquet = {bp: bp.rsplit(".", 1)[0] + ".parquet" for _, bp, _, _ in entries}
    m = build_file_identity_map(catalog, parquet, "c")
    ident = m.resolve_table("ORD")
    assert ident.coverage_start == "2022-01-01"
    assert ident.coverage_end == "2024-12-31"


def test_unauthorized_table_raises_authorization_error():
    catalog, parquet = _catalog()
    m = build_file_identity_map(catalog, parquet, "cont1")
    raised = None
    try:
        # authorize nothing → the (otherwise valid) table is unauthorized
        canonicalize_logical_sql("SELECT * FROM PROC_PURCHASEORDERS", m, allowed_file_ids=set())
    except SQLCanonicalizationError as exc:
        raised = exc
    assert isinstance(raised, SQLAuthorizationError)
    assert not isinstance(raised, SQLParseError)
