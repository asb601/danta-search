"""F2 — fan-out / cartesian-aggregation guard.

The proven failure (I11): SUM additive measures from BOTH sides of a join on a
non-unique key → each side fans the other → $86M reported as $2.5B
(= 38 rows x 29 rows). The high-precision, cardinality-free signal: additive
aggregates (SUM/AVG) pulling from >= 2 distinct base tables WITHIN one joined
SELECT scope. Pre-aggregated CTE joins and self-joins must pass.
"""
from app.services.execution_guards import check_fanout_risk


def test_flags_two_sided_sum_over_join():
    sql = (
        "SELECT p.VENDOR_ID, SUM(p.AMOUNT_LIMIT), SUM(i.INVOICE_AMOUNT) "
        "FROM PO_HEADERS_ALL p JOIN AP_INVOICE_DISTRIBUTIONS_ALL i "
        "ON p.VENDOR_ID = i.VENDOR_ID GROUP BY p.VENDOR_ID"
    )
    r = check_fanout_risk(sql)
    assert r.ok is False
    assert r.offending_tables is not None
    assert set(r.offending_tables) == {"PO_HEADERS_ALL", "AP_INVOICE_DISTRIBUTIONS_ALL"}


def test_no_join_single_table_is_ok():
    r = check_fanout_risk("SELECT SUM(INVOICE_AMOUNT) FROM AP_INVOICES_ALL")
    assert r.ok is True


def test_single_sided_aggregate_over_join_is_ok():
    # Aggregating the many-side (lines) of a header/line join is correct, not fan-out.
    sql = (
        "SELECT o.ORDER_NUMBER, SUM(l.LINE_TOTAL) "
        "FROM OE_ORDER_HEADERS_ALL o JOIN OE_ORDER_LINES_ALL l "
        "ON o.HEADER_ID = l.HEADER_ID GROUP BY o.ORDER_NUMBER"
    )
    assert check_fanout_risk(sql).ok is True


def test_preaggregated_cte_join_is_ok():
    # The CORRECT way to do the I11 query — each side pre-aggregated in its own
    # CTE, then joined. Must NOT be flagged.
    sql = (
        "WITH po AS (SELECT VENDOR_ID, SUM(AMOUNT_LIMIT) amt FROM PO_HEADERS_ALL GROUP BY VENDOR_ID), "
        "inv AS (SELECT VENDOR_ID, SUM(INVOICE_AMOUNT) inv FROM AP_INVOICE_DISTRIBUTIONS_ALL GROUP BY VENDOR_ID) "
        "SELECT po.VENDOR_ID, po.amt, inv.inv FROM po JOIN inv ON po.VENDOR_ID = inv.VENDOR_ID"
    )
    assert check_fanout_risk(sql).ok is True


def test_self_join_same_table_is_ok():
    # Manager rollup self-join (I15) — both summed columns are the SAME logical
    # table, so there is no cross-table fan-out.
    sql = (
        "SELECT a.SUPERVISOR_ID, SUM(a.ANNUAL_SALARY), SUM(b.ANNUAL_SALARY) "
        "FROM PER_ALL_PEOPLE_F a JOIN PER_ALL_PEOPLE_F b "
        "ON a.PERSON_ID = b.SUPERVISOR_ID GROUP BY a.SUPERVISOR_ID"
    )
    assert check_fanout_risk(sql).ok is True


def test_malformed_sql_fails_open():
    # The safety contract: the guard must NEVER block on SQL it can't analyse,
    # whether sqlglot rejects it (parse_ok False) or leniently parses junk.
    for junk in ("this is not sql ;;;", "SELECT FROM JOIN ((", ""):
        assert check_fanout_risk(junk).ok is True

