"""
System prompt builder — assembles the prompt from catalog data,
parquet paths, and conversation context.
"""
from __future__ import annotations

import re
from datetime import date, timedelta

from app.agent.state import MAX_TOOL_CALLS
from app.core.logger import chat_logger
from app.services.file_identity import FileIdentityMap, logical_name_from_path


# Auto-generated descriptions often start with absolutist phrases like
# "This file is the PRIMARY source for..." or "Unlike similar files, this
# file...". Those phrases over-anchor the LLM and stop it from considering
# alternative files in the catalog. We strip them at render time so the
# stored description is unchanged but the prompt sees neutral text.
_ANCHOR_PATTERNS = [
    re.compile(r"\bThis file is the PRIMARY source\b", re.IGNORECASE),
    re.compile(r"\bThis file is THE PRIMARY source\b", re.IGNORECASE),
    re.compile(r"\bPRIMARY source\b"),
    re.compile(r"\bUnlike (?:other|similar) files,?\s*", re.IGNORECASE),
    re.compile(r"\bnot (?:typically )?found in other (?:similar )?files\b", re.IGNORECASE),
]


def _neutralize_description(desc: str) -> str:
    """Remove over-anchoring phrases from auto-generated descriptions."""
    if not desc:
        return ""
    out = desc
    for pat in _ANCHOR_PATTERNS:
        out = pat.sub("", out)
    # Collapse double spaces and stray leading punctuation introduced by removals
    out = re.sub(r"\s{2,}", " ", out).strip()
    out = re.sub(r"^[,;:.\s]+", "", out)
    return out


SYSTEM_PROMPT_TEMPLATE = """{file_override_note}You are a data analyst with read-only SQL access to logical tables.

Runtime owns storage: never write blob paths, parquet filenames, physical storage
URIs, or storage scan functions. Use only the logical table names
shown below in FROM/JOIN clauses; the runtime resolves them to authorized files.
Each logical table already spans all of its time periods — query the table name
once; do NOT union per-month tables or assume a period is "missing" from one file.

SQL dialect: the executor is DuckDB. Write DuckDB-valid SQL. Use date_diff('day', a, b)
(not DATEDIFF), string_agg(x, ',') (not GROUP_CONCAT), and current_date. When a value
already exists as a column (e.g. an aging/days/DSO column), use that column directly
rather than recomputing it from current_date.

Reference date for relative time: {today_iso} ({today_human}) — the most recent
date this dataset covers. Resolve every relative time expression in the user's
question against THIS date (not the wall clock, not your training cutoff, and NOT
SQL current_date, which may be later than the data). Examples:
  - "last month"        → the full previous calendar month ({last_month_start} to {last_month_end})
  - "this month" / MTD  → {this_month_start} to {today_iso}
  - "YTD" / "this year"  → {year_start} to {today_iso}
  - "last year"         → {last_year_start} to {last_year_end}
  - "last 30 days"      → {last_30_start} to {today_iso}
For relative-time filters use these explicit dates, not current_date.
Never invent a date range from a year you remember from training data.

COUNTING & DISTINCTNESS: "how many / number of / distinct <entity>" means
COUNT(DISTINCT <entity key>). A tool result's total_rows is a ROW count — the
same entity can repeat across rows (e.g. one VENDOR_ID with several names) — so
never report total_rows as the count of distinct entities. To count entities,
COUNT(DISTINCT <id column>); a multi-column SELECT DISTINCT counts distinct row
combinations, not distinct entities.

Dataset scope: current authorized catalog.
{shortlist_header}
{parquet_note}
{sample_note}

--- TOOLS ---
1. run_sql             \u2014 Execute SQL against logical tables.
2. get_file_schema     \u2014 Returns column names, types, and sample values for a logical table.
3. inspect_column      \u2014 Returns dtype, sample values, and a suggested WHERE predicate for a
                        single column. Use before any filter whose storage format is unclear
                        (dates, codes, years, identifiers). Cheap — prefer over guessing.
4. search_catalog      \u2014 Searches the FULL catalog ({total_file_count} files) by metadata
                        (table names, descriptions, column names). It does NOT search row
                        values. Use when the shortlist doesn't clearly contain what you need.
5. inspect_data_format \u2014 Preview raw rows from a specific logical table.
6. summarise_dataframe \u2014 Compute stats on the last SQL result.
7. extract_relations   \u2014 Returns scoped join relationships and bounded multi-hop paths.
                        Pass the smallest set of tables you have selected. Start with
                        direct joins; request multi-hop only when selected files are not
                        directly connected.

--- ANALYST WORKFLOW ---
Work through every data request in four phases.

PHASE 1 \u2014 DECOMPOSE
Identify (a) the primary business subject and (b) each requested facet or section.
Anchor everything on the primary subject. Treat additional facets as evidence
requirements attached to that anchor, not as permission to explore unrelated domains.
For "analyze" or "summarize" requests: plan aggregate SQL from the start, not row
detail, unless the user explicitly asks to list records.

PHASE 2 \u2014 GROUND  (mandatory before writing any SQL)
Inspect schemas before assuming anything.
\u2022 Call get_file_schema on the primary shortlisted file(s) first.
\u2022 Call inspect_column for any column whose format is unclear.
\u2022 For each requested facet, check the schemas you just inspected: if the required
  column already exists there, use it directly. Only call search_catalog for a facet
  when no already-inspected schema contains the needed column.
Schema knowledge from previous queries is stale \u2014 always re-inspect in this request.

PHASE 3 \u2014 CONNECT  (only when the answer genuinely needs more than one file)
Call extract_relations with the smallest possible file set.
Use the returned join columns directly. Candidate/technical_candidate relationships
are unverified \u2014 inspect column values before joining on them. If no direct
relationship exists, request a bounded multi-hop path; if none is returned, join
manually and flag it as unverified in your response.

PHASE 4 \u2014 EXECUTE AND ADAPT
Write SQL using only column names verified in Phase 2, and only filter VALUES you
have actually observed (via inspect_column / get_file_schema). Never invent a
status/category literal (e.g. do not assume a value 'Shipped' exists).
\u2022 0 rows on an aggregation/filter is NOT permission to switch tables or domains.
  First, on the SAME table(s): (a) drop your most specific filter and re-run;
  (b) probe the date column with SELECT MIN(col), MAX(col) to confirm your window
  overlaps the data \u2014 the data may simply not cover the requested period; (c)
  SELECT DISTINCT the status/category column you filtered to confirm the literal
  exists. Only after these may you conclude the value/period is genuinely absent.
\u2022 Switching to a table from a DIFFERENT business domain after a 0-row aggregation
  is almost always wrong \u2014 do not join across unrelated domains to manufacture rows.
\u2022 JOIN fails \u2192 re-examine the join column with inspect_column.
\u2022 Column missing \u2192 search_catalog for an alternative table.
Never retry the same query with only cosmetic changes \u2014 change the approach.

TREND / DIVERGENCE QUESTIONS (one metric worsening while another rises, over time):
\u2022 Normalise both metrics to ONE period grain before comparing. If one table is at
  month/YYYYMM and another coarser, roll the finer up (quarter = 'Q' || CAST(CEIL(MONTH/3) AS INT)).
  Never compare metrics at mismatched grains.
\u2022 If the two metrics live in tables that do not share the comparison dimension, route
  the metric-bearing table through the bridge table named in WORKFLOW TOPOLOGY to acquire
  that dimension BEFORE grouping (e.g. a deliveries table lacking a region dimension joins
  through an orders table that carries both the plant and the region).
\u2022 Compute period-over-period change per dimension with a window:
  metric - LAG(metric) OVER (PARTITION BY <dim> ORDER BY <period>).
\u2022 "Divergence" = periods where the two deltas move in the worsening direction together;
  rank by joint magnitude to find the sharpest period. Do NOT answer with two independent
  aggregates \u2014 the question asks whether they track each other.

TWO CORE PRINCIPLES (apply across all phases)
A. Evidence is not transferable. Delivery status \u2260 approval status \u2260 payment status.
   Each facet requires its own data evidence even if the concepts sound related.
B. "No data found" is a last resort, not a first answer. Investigate before giving up.

--- QUESTION TYPE ---
Conceptual ("how does X work", "explain Y"): answer from knowledge + file
descriptions. No SQL needed unless you need a column list.
Data ("show me", "how many", "top N", filters): run SQL using the steps above.

--- OUTPUT STYLE (MANDATORY) ---
Do NOT narrate your reasoning, plans, or next steps (no \"Let me start by\u2026\",
\"Plan: 1. \u2026\", \"I'll now query\u2026\"). Reasoning happens silently via tool calls.

When you finish, write a complete analyst response:

1. **Direct answer** \u2014 one sentence that directly answers the question
    (e.g. \"The top 5 records by outstanding balance total $4.2M across
    312 open items.\").
2. **Key insights** \u2014 2\u20134 bullet points interpreting the data (patterns,
   outliers, comparisons, anything actionable). Write as a business analyst.
3. **Table note** — if SQL returned rows, end with the line:
   "↓ See the results table below for the full data."
4. **Source** — one short line stating which logical table(s) the data came from
   and the filter applied.

Do NOT include tabular data in the text — no markdown pipe tables, no CSV rows.
The UI renders the SQL results as an interactive table directly below this
response. Only state numeric totals that are explicitly in the result rows.

If you cannot answer, say so in one sentence and state which logical tables you checked.
Do not ask the user \"would you like me to search\u2026\" \u2014 just go search.

Max {max_calls} tool calls total.
"""


_DESC_MAX_CHARS = 200  # max characters shown per file description in the prompt
_DIM_METRIC_LIMIT = 4  # max key_dimensions / key_metrics shown per file
_PROMPT_COLUMN_LIMIT = 40  # exact columns shown for priority files


# ── SME join-enforcement prompt swap (flag-gated, byte-identical when off) ─────
# When relationship-graph join enforcement is active, the execution layer will
# REJECT any JOIN whose table pair is not an approved relationship. The default
# Phase-3 guidance below licenses the model to "join manually and flag it as
# unverified" — which directly contradicts that enforcement (the manual join is
# rejected, not flagged). When the flag is on we replace ONLY that sentence with
# the approved-path-or-independent instruction, leaving the rest of the prompt
# verbatim. When the flag is off, no substitution runs and the prompt is the
# unchanged template (byte-identical).
_JOIN_LICENSE_ORIGINAL = (
    "If no direct\n"
    "relationship exists, request a bounded multi-hop path; if none is returned, join\n"
    "manually and flag it as unverified in your response."
)
_JOIN_LICENSE_ENFORCED = (
    "If no direct\n"
    "relationship exists, request a bounded multi-hop path; if none is returned, do NOT\n"
    "join the tables — analyze each table independently and state that no validated\n"
    "relationship exists between them. A manual/invented join will be rejected at execution."
)


# ── OEBS dataset domain knowledge (static, injected at prompt tail) ───────────
# This block is appended to every system prompt when the dataset is the Ivy Data
# Company Oracle E-Business Suite (OEBS) corpus (~1,000 tables across Finance,
# Supply Chain, HR, and Sales Operations).  It gives the agent a deterministic
# table-routing map so it never has to guess which module owns a given concept.
_OEBS_DOMAIN_KNOWLEDGE = """
--- OEBS DATASET REFERENCE ---
Dataset: Ivy Data Company Oracle E-Business Suite (~1,000 tables).
Default operating unit filter: ORG_ID = 204 (apply to every query unless the
user explicitly requests all organizations).
Shared linking keys across the entire dataset:
  CUSTOMER_ID   → Sales (OE), Receivables (AR), Customer Profiles (HZ)
  VENDOR_ID     → Purchasing (PO), Payables (AP)
  ITEM_NUMBER   → Inventory (MTL), BOM, Costing (CST), Order Lines (OE)
  PERSON_ID     → HR People (PER), Assignments (PER), Payroll (PAY), Benefits (BEN)

FINANCE (GL · AP · AR · FA · CE · XLA · IBY · ZX · FND)
  GL_BALANCES               period net debits/credits, YTD balances, account type
  GL_JE_HEADERS / _LINES    journal entries; join headers→lines on JE_HEADER_ID
  GL_CODE_COMBINATIONS      chart of accounts; SEGMENT2 = cost center
  GL_DAILY_RATES            currency conversion rates by date
  AP_INVOICES_ALL           supplier invoices: amount, due date, payment status, tax
  AP_HOLDS_ALL              hold reasons; join to AP_INVOICES_ALL
  AP_CHECKS_ALL             payment checks by bank account
  AR_TRANSACTIONS_ALL       customer invoices, open balances, amount due remaining
  AR_CASH_RECEIPTS_ALL      cash received from customers, trend by month
  FA_ADDITIONS_B            asset master: cost, location, depreciation method, in-service date
  FA_DEPRN_SUMMARY          accumulated depreciation by asset category
  CE_BANK_ACCOUNTS / _STATEMENTS  bank balances and statement lines; flag uncleared
  XLA_EVENTS / XLA_AE_HEADERS    accounting events and entry headers
  IBY_PAYMENTS_ALL          payment records by method and amount
  ZX_LINES                  tax lines: recoverable vs non-recoverable, jurisdiction
  FND_LOOKUP_VALUES         lookup codes and descriptions (e.g. DOCUMENT_TYPE)
  Finance KPIs:
    Net Book Value    = FA_ADDITIONS_B.cost − FA_DEPRN_SUMMARY.deprn_reserve
    Overdue AP        = AP_INVOICES_ALL where due_date < today AND amount_remaining > 0
    Effective tax rate= ZX_LINES.tax_amount / AP_INVOICES_ALL.taxable_amount, by jurisdiction
    AR↔GL reconcile   = match AR_CASH_RECEIPTS_ALL to GL_JE_LINES where GL source = 'Receivables'
    FA↔GL reconcile   = match FA_DEPRN_SUMMARY to GL_JE_LINES by period

SUPPLY CHAIN (PO · RCV · MTL · BOM · WIP · CST · MRP · MSC · EAM)
  PO_HEADERS_ALL            PO headers: vendor, status, total amount
  PO_DISTRIBUTIONS_ALL      PO line distributions by department (attribute_category)
  RCV_TRANSACTIONS          receipt/inspection transactions; types: RECEIVE, INSPECT, REJECT, DELIVER
  MTL_MATERIAL_TRANSACTIONS material movements: issues, receipts, transfers; cost and qty by subinventory
  MTL_ONHAND_QUANTITIES_DETAIL  current on-hand stock by item and subinventory
  MTL_SYSTEM_ITEMS_B        item master: descriptions and attributes
  BOM_BILL_OF_MATERIALS / BOM_COMPONENTS_B  assembly headers and component quantities
  WIP_DISCRETE_JOBS         work orders: start qty, scrapped qty, job class, scheduled completion
  CST_ITEM_COSTS            unit cost by item and cost type (material + resource + overhead)
  MRP_RECOMMENDATIONS       planning exceptions including SHORTAGE type
  MSC_SUPPLIES / MSC_DEMANDS planned supply and demand quantities by item
  EAM_WORK_ORDERS           maintenance work orders; filter EMERGENCY type and open status
  Supply Chain KPIs:
    Scrap rate        = WIP scrapped_qty / WIP start_qty, per job class
    Under-shipment    = OE_ORDER_LINES_ALL.ordered_qty − WSH_DELIVERY_DETAILS.shipped_qty (where negative)
    Inventory gap     = MTL on-hand qty − MRP planned demand, per ITEM_NUMBER
    PO shortfall      = PO ordered qty − RCV received qty, join on PO_HEADER_ID

HUMAN RESOURCES (PER · PAY · BEN · OTA · IRC)
  PER_ALL_PEOPLE_F          employee master: name, grade, annual salary, location, employee type
  PER_ALL_ASSIGNMENTS_F     assignment history: grade, salary by date range
  PER_ABSENCE_ATTENDANCES   absence records; group by department
  PAY_RUN_RESULTS           payroll run results; join to PAY_ELEMENT_TYPES_F
  PAY_ELEMENT_TYPES_F       element names (earnings, deductions, net pay)
  PAY_RUN_RESULT_VALUES     individual element values per run
  PAY_COSTS                 payroll cost by month and cost center
  BEN_PRTT_ENRT_RSLT_F      employee benefit plan enrollments
  BEN_PRTT_PREM_F           benefit premium amounts by plan and month
  OTA_EVENTS / OTA_BOOKINGS training event catalog and employee bookings
  IRC_APPLICANTS / IRC_OFFERS / IRC_JOB_POSTINGS  recruitment pipeline
  HR KPIs:
    Active employees  = PER_ALL_PEOPLE_F where employee_type is active AND effective_end_date > today (or null)
    VP or above       = PER grade field containing VP, SVP, EVP, or C-level title
    Total compensation= PER annual salary + PAY net pay + BEN premiums, joined on PERSON_ID
    Headcount         = COUNT(DISTINCT PERSON_ID) for active assignments
    Absence rate      = absence count / headcount, by department

SALES OPERATIONS (OE · WSH · HZ · PA · CS · ASO · AS · QP · OKC · JTF)
  OE_ORDER_HEADERS_ALL      sales order headers: customer, status (BOOKED/CLOSED), order date
  OE_ORDER_LINES_ALL        order lines: item, ordered qty, unit price, line value
  WSH_DELIVERY_DETAILS      shipments: shipped qty, carrier, freight cost, actual vs scheduled ship date
  HZ_PARTIES                customer master: name, total invoiced amount
  HZ_CUSTOMER_PROFILES      credit rating and credit limit per customer
  PA_PROJECTS_ALL           project master: budget, actual cost, variance
  CS_SERVICE_REQUESTS       service incidents: priority (P1=Critical), status, CSAT score
  ASO_QUOTE_HEADERS_ALL     quote headers by status and total value
  AS_OPPORTUNITIES_ALL      pipeline: amount, win probability, sales stage, WON/LOST
  QP_LIST_LINES             price list lines: active status, discount percentage
  OKC_K_HEADERS_ALL_B       service contract headers: start date, end date, status
  JTF_TASKS_ALL_B           task records: owner, status (open/closed)
  Sales KPIs:
    Late shipment      = WSH actual_ship_date > WSH scheduled_ship_date
    Weighted pipeline  = AS opportunity_amount × win_probability_pct
    Over-budget project= PA actual_cost > PA budget_amount; variance = actual − budget
    Expiring contracts = OKC end_date within next 6 months from today
    High-risk customer = HZ credit_rating below BBB AND credit_limit > 1,000,000

CROSS-DOMAIN JOINS (use extract_relations first; fall back to these keys only when confirmed)
  PO→AP  (Procure-to-Pay)          VENDOR_ID
  OE→AR  (Order-to-Cash)           CUSTOMER_ID
  OE→WSH (Order Fulfillment)       ORDER_LINE_ID
  AR→GL  (Receivables recon)       GL source = 'Receivables'
  FA→GL  (Depreciation recon)      period match
  PA→AP  (Project Spend)           PROJECT_ID
  PER+PAY+BEN (Total Compensation) PERSON_ID
  MTL+MRP (Inventory vs Demand)    ITEM_NUMBER
  RCV+PO  (Receipt Shortfall)      PO_HEADER_ID
  CS+MTL  (Defect Analysis)        ITEM_NUMBER
  HZ→OE→AR (360 Customer View)     CUSTOMER_ID (chain)
  AP+ZX  (Tax Analysis)            INVOICE_ID

DISCOVERY RULE: if the user asks "which table contains X" or "where is Y data",
consult the catalog (search_catalog tool) first — the _DATASET_CATALOG.csv is
the authoritative index of all ~1,000 tables, their domains, row counts, and
key columns.
"""

# ── SAP ECC dataset domain knowledge (static, injected at prompt tail) ────────
# Ivy Data Company SAP ECC corpus (~1,000 tables, Client 100).
# Covers FI, CO, AA, TR, MM, WM, SD, HCM modules.
_SAP_DOMAIN_KNOWLEDGE = """
--- SAP ECC DATASET REFERENCE ---
Dataset: Ivy Data Company SAP ECC (~1,000 tables, Client MANDT=100).
Default filters: always add MANDT = '100'. Company codes: 1000=US, 2000=EU, 3000=APAC.
Controlling area: KOKRS = 1000. Plan version: RVERS/VERSN = 0 (actual).

SHARED JOIN KEYS (referential integrity across modules):
  BUKRS   Company Code        → FI, CO, AA, HCM, MM, SD, TR
  MATNR   Material Number     → MM, SD, CO, FI, WM
  KUNNR   Customer Number     → FI, SD, CO
  LIFNR   Vendor Number       → FI, MM, AA, SD
  PERNR   Personnel Number    → HCM, SD
  KOSTL   Cost Center         → CO, FI, AA, HCM, MM, SD
  PRCTR   Profit Center       → CO, FI, SD
  KOKRS   Controlling Area    → CO
  WERKS   Plant               → MM, SD, AA, CO, FI, HCM, WM
  LGORT   Storage Location    → MM, SD, WM
  VKORG   Sales Organization  → SD, FI
  VTWEG   Distribution Channel→ SD, FI
  SPART   Division            → SD, FI
  EKORG   Purchasing Org      → MM
  ANLN1   Main Asset Number   → AA
  EBELN   PO Number           → MM
  VBELN   SD Document Number  → SD
  BELNR   FI Document Number  → FI, CO, MM, SD
  GJAHR   Fiscal Year         → FI, CO, AA, MM
  SAKNR   G/L Account         → FI
  AUFNR   Internal Order      → CO, MM
  LGNUM   Warehouse Number    → WM
  WAERS   Currency            → all modules

FINANCE / FICO (Modules: FI · CO · AA · TR)
  FI — Financial Accounting:
    Key tables: BKPF (doc header), BSEG (line items), BSIS/BSAS (open/cleared items),
                SKA1 (G/L master), KNA1 (customer master), LFA1 (vendor master),
                FAGLFLEXT (new G/L totals), FAGLFLEXA (new G/L line items)
    Key fields: BELNR doc number · BUDAT posting date · BLDAT doc date · GJAHR year
                MONAT period · BLART doc type · SHKZG debit/credit · DMBTR local amt
                WRBTR doc currency amt · HKONT G/L account · MWSKZ tax code
    Filters: BSTAT=' ' for open items; SHKZG='S' debit / 'H' credit
  CO — Controlling:
    Key tables: CSKS (cost center master), CSKA (cost elements), COSS/COSP (order totals),
                COEP (CO line items), CE1xxxx (COPA — replace xxxx with operating concern)
    Key fields: KOSTL cost center · KSTAR cost element · AUFNR order · PRCTR profit center
                PERIO period · WRTTP value type (04=actual, 01=plan) · BEKNZ debit/credit
                OBJNR object number · VERSN version
  AA — Asset Accounting:
    Key tables: ANLA (asset master), ANLZ (time-dep data), ANLC (depreciation values), ANEK (doc header)
    Key fields: ANLN1 asset · ANLN2 sub-asset · ANLKL asset class · AFABE dep area
                NAFAZ ordinary depreciation · KANSW cumulative acquisition · AKTIV capitalization date
    KPI: Net Book Value = KANSW (acquisition) − KNAFA (accumulated depreciation)
  TR — Treasury:
    Key tables: VTBFHA (financial transactions), VTBBEWE (flows/cash flows)
    Key fields: RFHA deal number · PRODUCT_TYPE instrument type · PARTNER counterparty
                DEAL_DATE trade date · DFAEL due date · VALUT value date · BZBETR nominal amt

LOGISTICS / MM+WM (Modules: MM · WM)
  MM — Materials Management:
    Key tables: MARA (material master), MARC (plant data), MARD (storage loc stock),
                MBEW (valuation), MCHB (batch stock), EKKO (PO header), EKPO (PO lines),
                EBAN (purchase requisition), MSEG (material doc lines), MKPF (material doc header)
    Key fields: MATNR material · MTART type (ROH=raw,FERT=finished,HAWA=trading)
                BWART movement type (101=GR,261=issue,301=transfer) · MENGE qty
                NETPR net price · PEINH price unit · MEINS base UoM · LABST unrestricted stock
                BEDAT PO date · LIFNR vendor · BSART PO type (NB=standard,UB=stock transport)
                MAKTX material description · MATKL material group · STPRS standard price
                VERPR moving avg price · VPRSV price control (S=standard,V=moving avg)
  WM — Warehouse Management:
    Key tables: LTBK (transfer order header), LTAP (transfer order items), LQUA (quants)
    Key fields: LGNUM warehouse · LGTYP storage type · TANUM TO number · BWLVS WM movement type
                VLTYP/VLPLA source type/bin · NLTYP/NLPLA dest type/bin

SALES / SD (Module: SD)
  Key tables: VBAK (order header), VBAP (order lines), VBUK/VBUP (status),
              LIKP (delivery header), LIPS (delivery lines),
              VBRK (billing header), VBRP (billing lines), VBFA (document flow)
  Key fields: VBELN document · AUART order type · VKORG sales org · VTWEG channel
              KUNNR/KUNAG sold-to · MATNR material · POSNR item · KWMENG confirmed qty
              NETWR net value · WAERK currency · FKART billing type · FKDAT billing date
              GBSTK overall status · LFSTK delivery status · FKSTK billing status
  SD KPIs:
    Late delivery  = LIKP actual GI date (WADAT_IST) > LIKP planned GI date (WADAT)
    Open orders    = VBAK where GBSTK not 'C' (closed)
    Revenue        = sum VBRP.NETWR where FKSTO <> 'X' (not cancelled)

HUMAN CAPITAL / HCM (Module: HCM)
  Key tables: PA0001 (org assignment), PA0002 (personal data), PA0008 (basic pay),
              PA0014 (recurring payments/deductions), PA0007 (planned working time),
              T549Q (payroll periods), PCL1/PCL2 (payroll result clusters)
  Key fields: PERNR employee · BEGDA/ENDDA validity dates · BUKRS company code
              WERKS plant/location · KOSTL cost center · ORGEH org unit · PLANS position
              ANSAL annual salary · LGART wage type · TRFGR pay scale group · TRFST level
              PERSG employee group · PERSK employee subgroup · EMPCT employment %
              ABWTG absence days · AWART absence/attendance type
  Filters: active employees = ENDDA >= today AND BEGDA <= today on PA0001
  HCM KPIs:
    Headcount      = COUNT(DISTINCT PERNR) on active PA0001 records
    Absence rate   = SUM(ABWTG) / headcount by ORGEH (org unit)
    Total pay      = SUM of BETRG on PA0008 + PA0014 joined on PERNR

CROSS-MODULE JOIN PATTERNS (confirm with extract_relations first):
  PO → GR → Invoice (Procure-to-Pay): EKKO/EKPO → MSEG (EBELN) → BSEG (BELNR)
  SD Order → Delivery → Billing:       VBAK → LIKP → VBRK via VBFA (document flow on VBELN)
  CO → FI reconciliation:              COEP.BELNR = BKPF.BELNR (same accounting document)
  Asset → FI:                          ANEK.BELNR = BKPF.BELNR
  HCM payroll → CO cost:               PA cost postings via KOSTL (cost center)
  Material valuation → FI:             MBEW.MATNR+BWKEY joins to BSEG via movement docs

FIELD NAME TRANSLATION (user says → SAP field):
  "company"/"entity"        → BUKRS    "vendor"/"supplier"    → LIFNR
  "customer"/"sold-to"      → KUNNR    "material"/"product"   → MATNR
  "employee"/"person"       → PERNR    "cost center"          → KOSTL
  "profit center"           → PRCTR    "plant"                → WERKS
  "PO number"               → EBELN    "sales order"          → VBELN (VBAK)
  "invoice" (AP)            → BELNR    "billing doc"          → VBELN (VBRK)
  "posting date"            → BUDAT    "document date"        → BLDAT
  "fiscal year"             → GJAHR    "period"               → MONAT/PERIO
  "debit"                   → SHKZG='S'"credit"               → SHKZG='H'
  "actual cost"             → WRTTP='04'"plan cost"           → WRTTP='01'
  "unrestricted stock"      → LABST    "standard price"       → STPRS
  "moving avg price"        → VERPR    "annual salary"        → ANSAL
  "absence days"            → ABWTG    "wage type"            → LGART
"""


def _column_names_for_prompt(entry: dict | None) -> list[str]:
    if not entry:
        return []
    names: list[str] = []
    for col in entry.get("columns_info") or []:
        if isinstance(col, dict) and col.get("name"):
            names.append(str(col["name"]))
        elif isinstance(col, str):
            names.append(col)
    if not names:
        names = [str(c) for c in (entry.get("column_names") or []) if isinstance(c, str)]
    return names[:_PROMPT_COLUMN_LIMIT]


def build_parquet_note(
    catalog: list[dict],
    parquet_paths_all: dict[str, str],
    parquet_blob_path: str | None,
    container_name: str,
    *,
    top_blob_paths: set[str] | None = None,
    file_identities: FileIdentityMap | None = None,
) -> str:
    """Build the file-listing section of the system prompt.

    top_blob_paths: blobs that receive full column_stats context (top-N by
    retrieval score). All other files get a compact description to reduce
    total token load without removing any file from the shortlist.
    """
    catalog_by_blob: dict[str, dict] = {}
    for entry in catalog:
        bp = entry.get("blob_path")
        if bp:
            catalog_by_blob[bp] = entry

    if parquet_paths_all:
        lines = []
        seen_logical: set[str] = set()
        for blob, pq in parquet_paths_all.items():
            entry = catalog_by_blob.get(blob)
            identity = file_identities.identity_for_blob(blob) if file_identities else None
            logical_table = identity.sql_name if identity else logical_name_from_path(blob)
            # Consolidation: a logical table spans many partition blobs. Emit ONE
            # line per logical table (not per month) so the model sees a single
            # name with the TRUE coverage span — never 36 lines with conflicting
            # single-month ranges (which previously triggered false "missing" claims).
            if logical_table in seen_logical:
                continue
            seen_logical.add(logical_table)
            line = f"  {logical_table}"
            if identity and identity.sql_name != identity.logical_name:
                line += f"  (display: {identity.logical_name})"
            if identity and identity.partition_count > 1:
                line += f"\n    Partitions: {identity.partition_count} (one logical table; query the name once)"

            desc = _neutralize_description(entry.get("ai_description") if entry else "")
            if desc:
                # Truncate description to keep per-file token cost bounded.
                if len(desc) > _DESC_MAX_CHARS:
                    desc = desc[:_DESC_MAX_CHARS].rsplit(" ", 1)[0] + "…"
                line += f"\n    Description: {desc}"
            key_dimensions = (entry.get("key_dimensions") or []) if entry else []
            if key_dimensions:
                line += f"\n    Key dimensions: {', '.join(key_dimensions[:_DIM_METRIC_LIMIT])}"
            key_metrics = (entry.get("key_metrics") or []) if entry else []
            if key_metrics:
                line += f"\n    Key metrics: {', '.join(key_metrics[:_DIM_METRIC_LIMIT])}"

            # Surface the AGGREGATE coverage window across all partitions (from
            # the consolidated identity) so the model knows the full span, not one
            # month. Fall back to the single entry's range when no identity.
            if identity and (identity.coverage_start or identity.coverage_end):
                dr_start, dr_end = identity.coverage_start, identity.coverage_end
            else:
                dr_start = entry.get("date_range_start") if entry else None
                dr_end = entry.get("date_range_end") if entry else None
            if dr_start or dr_end:
                line += f"\n    Date range (full coverage): {dr_start or '?'} \u2192 {dr_end or '?'}"

            # Surface column stats only for top-retrieved files to keep prompt
            # token load bounded. Lower-ranked files get date range only.
            _is_priority = top_blob_paths is None or blob in top_blob_paths
            if _is_priority:
                columns = _column_names_for_prompt(entry)
                if columns:
                    line += f"\n    Available columns: {', '.join(columns)}"

            if _is_priority:
                _DATE_HINTS = ("year", "date", "period", "month", "fiscal", "quarter", "fy")
                col_stats = (entry.get("column_stats") or {}) if entry else {}
                range_parts = []
                for col_name, stats in col_stats.items():
                    if stats.get("dtype") == "numeric" and any(
                        h in col_name.lower() for h in _DATE_HINTS
                    ):
                        mn, mx = stats.get("min"), stats.get("max")
                        if mn is not None and mx is not None:
                            range_parts.append(f"{col_name}: {mn}\u2013{mx}")
                if range_parts:
                    line += f"\n    Column ranges: {', '.join(range_parts[:4])}"

            lines.append(line)

        note = (
            "Initial shortlist of likely logical tables:\n"
            + "\n".join(lines)
            + "\nUse these names directly in SQL FROM/JOIN clauses. Parquet/CSV storage is resolved internally."
        )

        # Also list CSV-only files (no parquet conversion)
        csv_only = [e for e in catalog if e.get("blob_path") and e["blob_path"] not in parquet_paths_all]
        if csv_only:
            csv_lines = []
            for entry in csv_only:
                bp = entry["blob_path"]
                identity = file_identities.identity_for_blob(bp) if file_identities else None
                logical_table = identity.sql_name if identity else logical_name_from_path(bp)
                if logical_table in seen_logical:
                    continue
                seen_logical.add(logical_table)
                csv_line = f"  {logical_table}"
                desc = _neutralize_description(entry.get("ai_description") or "")
                if desc:
                    csv_line += f"\n    Description: {desc}"
                # Note: leave key_dimensions / key_metrics intact below.
                key_dimensions = entry.get("key_dimensions") or []
                if key_dimensions:
                    csv_line += f"\n    Key dimensions: {', '.join(key_dimensions[:6])}"
                key_metrics = entry.get("key_metrics") or []
                if key_metrics:
                    csv_line += f"\n    Key metrics: {', '.join(key_metrics[:6])}"
                csv_lines.append(csv_line)
            note += (
                "\n\nCSV-only logical tables (runtime may execute them more slowly):\n"
                + "\n".join(csv_lines)
            )
        return note

    if parquet_blob_path:
        return (
            "Logical table access is available for the selected data. "
            "Use table names from search_catalog or get_file_schema; runtime resolves storage internally."
        )

    return ""


_CONV_CONTEXT_MAX_CHARS = 2000  # cap conversation history to bound token growth


def build_system_prompt(
    catalog: list[dict],
    parquet_paths_all: dict[str, str],
    parquet_blob_path: str | None,
    container_name: str,
    sample_rows_by_blob: dict[str, list],
    conversation_context: str = "",
    total_file_count: int | None = None,
    mentioned_files: list[str] | None = None,
    sql_context_note: str = "",
    *,
    top_blob_paths: set[str] | None = None,
    workflow_topology_note: str = "",
    file_identities: FileIdentityMap | None = None,
    as_of_date: date | None = None,
    erp_domain: str | None = None,
) -> str:
    """Assemble the full system prompt for the agent.

    as_of_date — the data-driven reference 'now' for relative-time resolution
    (the dataset's latest coverage date). Falls back to the wall clock only when
    the catalog carries no date coverage. See resolve_as_of_date().
    """
    parquet_note = build_parquet_note(
        catalog, parquet_paths_all, parquet_blob_path, container_name,
        top_blob_paths=top_blob_paths,
        file_identities=file_identities,
    )

    sample_note = ""
    if sample_rows_by_blob:
        sample_note = (
            f"\nData format preview: ingest-time example rows are available for {len(sample_rows_by_blob)} files via"
            " inspect_data_format(logical_table, n=5) — use this only after you know which table you want to inspect."
        )

    shortlist_count = len(catalog)
    full_count = total_file_count if total_file_count is not None else shortlist_count
    if full_count > shortlist_count:
        shortlist_header = (
            f"Showing the top {shortlist_count} of {full_count} ingested files "
            f"(retrieval-ranked for this query). The other "
            f"{full_count - shortlist_count} files are NOT shown — call "
            f"search_catalog to reach them."
        )
    else:
        shortlist_header = f"All {full_count} ingested files are shown below."

    today = as_of_date or date.today()
    # Last calendar month bounds
    first_of_this_month = today.replace(day=1)
    last_month_end = first_of_this_month - timedelta(days=1)
    last_month_start = last_month_end.replace(day=1)
    # Last calendar year bounds
    last_year_start = date(today.year - 1, 1, 1)
    last_year_end = date(today.year - 1, 12, 31)
    # Last 30 days
    last_30_start = today - timedelta(days=30)

    if mentioned_files:
        names = ", ".join(f"`{f}`" for f in mentioned_files)
        file_override_note = (
            f"USER SPECIFIED FILE: {names}\n"
            f"Query ONLY this file. Do not redirect to a different file based on "
            f"semantic matching. Call get_file_schema on {names} first, then run logical SQL on it.\n\n"
        )
    else:
        file_override_note = ""

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        container_name=container_name,
        max_calls=MAX_TOOL_CALLS,
        parquet_note=parquet_note,
        sample_note=sample_note,
        shortlist_header=shortlist_header,
        shortlist_count=shortlist_count,
        total_file_count=full_count,
        file_override_note=file_override_note,
        today_iso=today.isoformat(),
        today_human=today.strftime("%A, %d %B %Y"),
        this_month_start=first_of_this_month.isoformat(),
        last_month_start=last_month_start.isoformat(),
        last_month_end=last_month_end.isoformat(),
        year_start=date(today.year, 1, 1).isoformat(),
        last_year_start=last_year_start.isoformat(),
        last_year_end=last_year_end.isoformat(),
        last_30_start=last_30_start.isoformat(),
    )

    # SME join enforcement: when joins are rejected at execution, the prompt must
    # not license a manual/unverified join. Swap that one sentence (flag-gated).
    # Default OFF → no substitution → byte-identical prompt.
    try:
        from app.core.config import get_settings as _gs  # noqa: PLC0415
        _s = _gs()
        if (
            getattr(_s, "SME_MODE_ENABLED", False)
            and getattr(_s, "SME_JOIN_ENFORCE_ENABLED", False)
            and _JOIN_LICENSE_ORIGINAL in system_prompt
        ):
            system_prompt = system_prompt.replace(
                _JOIN_LICENSE_ORIGINAL, _JOIN_LICENSE_ENFORCED, 1
            )
    except Exception as _exc:  # never let prompt assembly fail on a flag read
        chat_logger.warning("sme_join_prompt_swap_error", error=str(_exc)[:200])

    # Inject validated SQL context right before the HOW TO WORK behavioural rules
    # so the LLM reads its constraints alongside its work instructions.
    # Workflow topology note (reachable joins + orphaned files) is injected
    # immediately after the SQL context block so the planner sees both together.
    _context_block = ""
    if sql_context_note:
        _context_block = sql_context_note
    if workflow_topology_note:
        _context_block = "\n\n".join(filter(None, [_context_block, workflow_topology_note]))
    if _context_block:
        _marker = "--- HOW TO WORK ---"
        if _marker in system_prompt:
            system_prompt = system_prompt.replace(
                _marker, _context_block + "\n\n" + _marker, 1
            )
        else:
            system_prompt += "\n\n" + _context_block

    if conversation_context:
        # Truncate conversation history to bound per-request token cost.
        # Long conversations grow linearly; most context is in the last few turns.
        _ctx = conversation_context
        if len(_ctx) > _CONV_CONTEXT_MAX_CHARS:
            _ctx = _ctx[-_CONV_CONTEXT_MAX_CHARS:]
            # Avoid starting mid-word after truncation
            nl = _ctx.find("\n")
            if 0 < nl < 200:
                _ctx = _ctx[nl + 1:]
        system_prompt += (
            "\n\n--- CONVERSATION HISTORY ---\n"
            "The user is continuing a conversation. Use this context to understand "
            "follow-up questions, pronouns ('it', 'that', 'those'), and references "
            "to previous queries or results.\n\n"
            f"{_ctx}\n"
            "---\n"
        )

    _domain_upper = (erp_domain or "").upper()
    if not _domain_upper or "ORACLE" in _domain_upper or "OEBS" in _domain_upper:
        system_prompt += _OEBS_DOMAIN_KNOWLEDGE
    if not _domain_upper or "SAP" in _domain_upper:
        system_prompt += _SAP_DOMAIN_KNOWLEDGE

    chat_logger.info("system_prompt_size",
                     chars=len(system_prompt),
                     words=len(system_prompt.split()),
                     parquet_file_count=len(parquet_paths_all),
                     has_conversation_context=bool(conversation_context))

    return system_prompt


# ── Entity extraction prompt ──────────────────────────────────────────────────

def build_entity_extraction_prompt(query: str) -> str:
    """
    Build the user message for GPT-4o-mini entity extraction.

    No system message — the instruction is embedded directly in the user turn
    to keep the call minimal (matches the llm_tasks single-message convention).
    Output contract: strict JSON {"entities": ["snake_case_concept", ...]}.
    """
    return (
        "Extract the business objects, processes, workflow states, exceptions, and "
        "relationships that a data agent needs to find the right tables for this query.\n"
        'Return ONLY valid JSON: {"entities": ["snake_case_concept"]}.\n'
        "Rules:\n"
        "1. Expand abbreviations (PO → purchase_order, SO → sales_order, etc.).\n"
        "2. Include workflow states, exceptions, matching/reconciliation, holds, and "
        "lifecycle events from colon sections and bullet lists when they need their own data.\n"
        "3. Anchor generic labels to their owner: po_approval_status not approval_status.\n"
        "4. Exclude: time ranges, metrics/values, display fields, recommendations, "
        "next actions, and output instructions.\n"
        "5. Return up to 10 concise singular snake_case entities.\n\n"
        f"Query: {query}"
    )
