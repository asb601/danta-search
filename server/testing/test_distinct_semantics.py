"""F3 — distinct-vs-row semantics note.

I02: 'list distinct VENDOR_NAME + VENDOR_ID' ran SELECT DISTINCT name, id → 2,422
distinct (name,id) TUPLES (dirty data: one id → many names) reported as '2,422
distinct vendors' (true 200). A multi-column SELECT DISTINCT's row count is the
count of distinct tuples, NOT distinct values of any single column.
"""
from app.services.execution_guards import distinct_projection_note


def test_multi_column_distinct_gets_note():
    note = distinct_projection_note(
        "SELECT DISTINCT VENDOR_NAME, VENDOR_ID FROM PO_VENDORS"
    )
    assert note and "COUNT(DISTINCT" in note


def test_single_column_distinct_is_clean():
    # SELECT DISTINCT one column → row count IS the distinct-entity count. No note.
    assert distinct_projection_note("SELECT DISTINCT VENDOR_ID FROM PO_VENDORS") is None


def test_count_distinct_is_clean():
    assert distinct_projection_note(
        "SELECT COUNT(DISTINCT VENDOR_ID) FROM PO_VENDORS"
    ) is None


def test_plain_select_is_clean():
    assert distinct_projection_note(
        "SELECT VENDOR_ID, INVOICE_AMOUNT FROM AP_INVOICES_ALL"
    ) is None


def test_unparseable_is_none():
    assert distinct_projection_note("not sql ;;;((") is None
    assert distinct_projection_note("") is None
