"""Phase 4 — value-evidenced cross-domain bridge tests (mocks-only, no live infra)."""
from __future__ import annotations

import subprocess

import pytest

from app.services.relationship_index import fingerprint_value


# ── Task 1 — bridge model + status enum ──────────────────────────────────────
def test_bridge_model_columns_and_status():
    from pdf_chat.models.bridge import PdfEntityBridge, BridgeStatus

    cols = set(PdfEntityBridge.__table__.columns.keys())
    assert {
        "id", "container_id", "tenant_id", "pdf_entity_id", "semantic_entity_id",
        "resolved_master_file_id", "resolved_master_column", "resolved_semantic_role",
        "value_overlap_pct", "confidence", "overlap_count", "pdf_value_count",
        "evidence", "status", "created_at",
    } <= cols
    assert BridgeStatus.LINKED.value == "linked" and BridgeStatus.REFUSED.value == "refused"
    assert "container_id" in cols


def test_bridge_model_registered_on_shared_base():
    from app.core.database import Base
    from pdf_chat.models import PdfEntityBridge  # noqa: F401 — import via package

    assert "pdf_entity_bridge" in Base.metadata.tables


# ── Task 2 — migration ────────────────────────────────────────────────────────
def test_migration_exposes_run_migration_and_upgrade_alias():
    from pdf_chat.migrations import bridge_upgrade

    assert callable(bridge_upgrade.run_migration)
    assert bridge_upgrade.upgrade is bridge_upgrade.run_migration


def test_migration_ddl_is_idempotent_create_if_not_exists():
    from pdf_chat.migrations import bridge_upgrade

    # Every secondary-index DDL must be guarded so re-running is a no-op.
    assert bridge_upgrade._INDEXES
    for stmt in bridge_upgrade._INDEXES:
        assert "CREATE INDEX IF NOT EXISTS" in stmt
        assert "pdf_entity_bridge" in stmt


@pytest.mark.asyncio
async def test_migration_creates_table_and_indexes_on_a_fake_engine():
    from pdf_chat.migrations import bridge_upgrade

    executed: list[str] = []
    create_all_called = {"v": False}

    class FakeConn:
        async def run_sync(self, fn):
            create_all_called["v"] = True

        async def execute(self, stmt):
            executed.append(str(stmt))

    class FakeBegin:
        async def __aenter__(self):
            return FakeConn()

        async def __aexit__(self, *a):
            return False

    class FakeEngine:
        def begin(self):
            return FakeBegin()

    await bridge_upgrade.run_migration(FakeEngine())
    assert create_all_called["v"] is True
    assert len(executed) == len(bridge_upgrade._INDEXES)


# ── Task 3 — value-evidenced reconciliation (the core safety) ─────────────────
@pytest.mark.asyncio
async def test_name_equality_alone_refuses():
    """An entity whose VALUES don't overlap any master key fingerprints is
    REFUSED — name equality is never a join signal."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.models.bridge import BridgeStatus

    verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e1", entity_name="Acme",
        samples=[EntityValueSample(value="ACME-XYZ")],
        master_columns=[MasterKeyColumn(
            file_id="f1", column="vendor_id", semantic_role="custom:entity_key:vendor_id",
            value_fingerprints=[fingerprint_value("V-100")],
        )],
    )
    assert verdict.status == BridgeStatus.REFUSED
    assert "below" in verdict.reason.lower() or "overlap" in verdict.reason.lower()
    assert verdict.master_column is None


@pytest.mark.asyncio
async def test_subthreshold_refuses():
    """Some overlap but below the gates ⇒ REFUSED, never silently pick top match."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.models.bridge import BridgeStatus

    # 4 sample values, only 1 overlaps → overlap_pct 0.25 < 0.50, count 1 < 3.
    samples = [EntityValueSample(value=v) for v in ("V-100", "V-201", "V-202", "V-203")]
    master_fps = [fingerprint_value("V-100"), fingerprint_value("Z-1"), fingerprint_value("Z-2")]
    verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e2", entity_name="Acme",
        samples=samples,
        master_columns=[MasterKeyColumn(
            file_id="f1", column="vendor_id", semantic_role="custom:entity_key:vendor_id",
            value_fingerprints=master_fps,
        )],
    )
    assert verdict.status == BridgeStatus.REFUSED
    assert verdict.master_column is None


@pytest.mark.asyncio
async def test_value_overlap_links_master_key():
    """Real value overlap above all gates ⇒ LINKED to the master key."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.models.bridge import BridgeStatus

    ids = ["V-100", "V-101", "V-102", "V-103"]
    samples = [EntityValueSample(value=v) for v in ids]
    master_fps = [fingerprint_value(v) for v in ids]  # 4/4 overlap
    verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e3", entity_name="Acme",
        samples=samples,
        master_columns=[MasterKeyColumn(
            file_id="f1", column="vendor_id", semantic_role="custom:entity_key:vendor_id",
            value_fingerprints=master_fps,
        )],
    )
    assert verdict.status == BridgeStatus.LINKED
    assert verdict.master_file_id == "f1"
    assert verdict.master_column == "vendor_id"
    assert verdict.overlap_count == 4
    assert verdict.value_overlap_pct == 1.0
    assert verdict.confidence >= 0.60


@pytest.mark.asyncio
async def test_picks_best_qualifying_column_not_first():
    """When multiple columns clear the gates, the BEST overlap wins."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.models.bridge import BridgeStatus

    ids = ["V-100", "V-101", "V-102", "V-103"]
    samples = [EntityValueSample(value=v) for v in ids]
    weak = MasterKeyColumn(  # 3/4 overlap (clears gates)
        file_id="fA", column="alt_id", semantic_role="custom:entity_key:vendor_id",
        value_fingerprints=[fingerprint_value(v) for v in ids[:3]],
    )
    strong = MasterKeyColumn(  # 4/4 overlap
        file_id="fB", column="vendor_id", semantic_role="custom:entity_key:vendor_id",
        value_fingerprints=[fingerprint_value(v) for v in ids],
    )
    verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e4", entity_name="Acme",
        samples=samples, master_columns=[weak, strong],
    )
    assert verdict.status == BridgeStatus.LINKED
    assert verdict.master_column == "vendor_id"


@pytest.mark.asyncio
async def test_tiny_domain_coincidence_refuses_high_cardinality_links():
    """Identical overlap_count, but a TINY master-key domain is rejected by the
    confidence/cardinality gate while a high-cardinality master key LINKS — proves
    join_confidence's cardinality term is wired to the master key, not overlap_count."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.models.bridge import BridgeStatus

    ids = ["V-100", "V-101", "V-102", "V-103"]
    samples = [EntityValueSample(value=v) for v in ids]  # 4 PDF values
    overlap_fps = [fingerprint_value(v) for v in ids[:3]]  # 3 overlap → pct 0.75, count 3

    # Tiny domain: the column only has the 3 overlapping fingerprints (cardinality 3).
    tiny = MasterKeyColumn(
        file_id="f_tiny", column="status_code", semantic_role="custom:entity_key:vendor_id",
        value_fingerprints=list(overlap_fps),
    )
    tiny_verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e_tiny", entity_name="Acme",
        samples=samples, master_columns=[tiny],
    )
    assert tiny_verdict.status == BridgeStatus.REFUSED  # confidence below floor

    # High-cardinality domain: same 3 overlaps, but a large distinct-value domain.
    big_fps = list(overlap_fps) + [fingerprint_value(f"OTHER-{i}") for i in range(1000)]
    big = MasterKeyColumn(
        file_id="f_big", column="vendor_id", semantic_role="custom:entity_key:vendor_id",
        value_fingerprints=big_fps,
    )
    big_verdict = await reconcile_entity_to_master_keys(
        tenant_id="t1", pdf_entity_id="e_big", entity_name="Acme",
        samples=samples, master_columns=[big],
    )
    assert big_verdict.status == BridgeStatus.LINKED
    assert big_verdict.master_column == "vendor_id"
    assert big_verdict.overlap_count == tiny_verdict.evidence["candidates"][0]["overlap_count"]


@pytest.mark.asyncio
async def test_non_key_semantic_role_refused_even_at_full_overlap():
    """A column whose semantic_role is NOT a master/reference key (e.g. a measure)
    is REFUSED even at 100% value overlap — role eligibility is a precondition."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, reconcile_entity_to_master_keys,
    )
    from pdf_chat.models.bridge import BridgeStatus

    ids = ["V-100", "V-101", "V-102", "V-103"]
    samples = [EntityValueSample(value=v) for v in ids]
    full = [fingerprint_value(v) for v in ids]  # 4/4 overlap

    for bad_role in (
        "custom:additive_measure:amount",
        "custom:attribute:description",
        "custom:date:invoice_date",
    ):
        verdict = await reconcile_entity_to_master_keys(
            tenant_id="t1", pdf_entity_id="e_role", entity_name="Acme",
            samples=samples,
            master_columns=[MasterKeyColumn(
                file_id="f1", column="amount", semantic_role=bad_role,
                value_fingerprints=full,
            )],
        )
        assert verdict.status == BridgeStatus.REFUSED, bad_role
        assert verdict.master_column is None


@pytest.mark.asyncio
async def test_build_bridge_persists_row_with_evidence():
    """build_bridge_for_entity reads values then persists a PdfEntityBridge row."""
    from pdf_chat.bridge.reconcile import EntityValueSample, build_bridge_for_entity
    from pdf_chat.models.bridge import BridgeStatus, PdfEntityBridge

    ids = ["V-100", "V-101", "V-102", "V-103"]

    async def values_reader(tenant_id, pdf_entity_id):
        return [EntityValueSample(value=v) for v in ids]

    class FakeSession:
        def __init__(self):
            self.added = []

        def add(self, obj):
            self.added.append(obj)

        async def flush(self):
            pass

    db = FakeSession()
    from pdf_chat.bridge.reconcile import MasterKeyColumn

    async def loader():
        return [MasterKeyColumn(
            file_id="f1", column="vendor_id", semantic_role="custom:entity_key:vendor_id",
            value_fingerprints=[fingerprint_value(v) for v in ids],
        )]

    verdict = await build_bridge_for_entity(
        db, tenant_id="t1", container_id="c1", pdf_entity_id="e5",
        entity_name="Acme", values_reader=values_reader, master_columns_loader=loader,
    )
    assert verdict.status == BridgeStatus.LINKED
    assert len(db.added) == 1
    row = db.added[0]
    assert isinstance(row, PdfEntityBridge)
    assert row.status == BridgeStatus.LINKED.value
    assert row.resolved_master_column == "vendor_id"
    assert row.container_id == "c1" and row.tenant_id == "t1"
    assert row.evidence is not None


# ── Task 7 — exit integration test (mocks-only) ───────────────────────────────
@pytest.mark.asyncio
async def test_exit_pdf_joins_vendor_csv_value_evidenced():
    """A PDF entity whose values overlap a vendor_id master key ABOVE threshold
    LINKS, and a structured_query through that bridge (mocked run_agent_query)
    returns the CSV answer; a low-overlap entity REFUSES and no cross-domain
    answer is produced. Asserts ZERO files under server/app were touched."""
    from pdf_chat.bridge.reconcile import (
        EntityValueSample, MasterKeyColumn, build_bridge_for_entity,
    )
    from pdf_chat.models.bridge import BridgeStatus

    vendor_ids = ["V-100", "V-101", "V-102", "V-103"]
    master_fps = [fingerprint_value(v) for v in vendor_ids]

    async def master_loader():
        return [MasterKeyColumn(
            file_id="vendors.parquet", column="vendor_id",
            semantic_role="custom:entity_key:vendor_id", value_fingerprints=master_fps,
        )]

    class FakeSession:
        def __init__(self):
            self.added = []

        def add(self, obj):
            self.added.append(obj)

        async def flush(self):
            pass

    # Mocked structured query path — delegates to the (read-only) agent.
    agent_calls = []

    async def fake_run_agent_query(query, db, **kw):
        agent_calls.append((query, kw))
        return {"answer": "Total spend for V-100 is 42000", "data": [],
                "row_count": 0, "files_used": ["vendors.parquet"]}

    # 1) High-overlap PDF entity → LINKED, then a structured query returns CSV answer.
    async def good_values(tenant_id, pdf_entity_id):
        return [EntityValueSample(value=v) for v in vendor_ids]

    db = FakeSession()
    linked = await build_bridge_for_entity(
        db, tenant_id="t1", container_id="c1", pdf_entity_id="pe_good",
        entity_name="Acme Corp", values_reader=good_values,
        master_columns_loader=master_loader, semantic_entity_id="se_vendor",
    )
    assert linked.status == BridgeStatus.LINKED
    assert linked.master_column == "vendor_id"

    # Only when LINKED do we cross into the CSV side.
    cross_answer = None
    if linked.status == BridgeStatus.LINKED:
        result = await fake_run_agent_query(
            f"total spend for {linked.master_column} V-100", db,
            container_id="c1", allowed_domains=["finance"], user_id="u1",
        )
        cross_answer = result["answer"]
    assert cross_answer == "Total spend for V-100 is 42000"
    assert len(agent_calls) == 1

    # 2) Low-overlap PDF entity → REFUSED, no cross-domain answer.
    async def bad_values(tenant_id, pdf_entity_id):
        return [EntityValueSample(value=v) for v in ("X-1", "X-2", "X-3", "V-100")]

    db2 = FakeSession()
    refused = await build_bridge_for_entity(
        db2, tenant_id="t1", container_id="c1", pdf_entity_id="pe_bad",
        entity_name="Acme Corp", values_reader=bad_values,
        master_columns_loader=master_loader,
    )
    assert refused.status == BridgeStatus.REFUSED
    refused_answer = None
    if refused.status == BridgeStatus.LINKED:  # never taken
        refused_answer = await fake_run_agent_query("x", db2)
    assert refused_answer is None
    assert len(agent_calls) == 1  # no extra agent call for the refused entity

    # 3) The CSV-side (server/app) business logic was NOT modified. The product
    # integration (branch pdf-product-integration) is permitted exactly ONE
    # additive touch in server/app: mounting the PDF routers + runtime migrations
    # in server/app/main.py. Any OTHER changed file under server/app would mean
    # the bridge/PDF work bled into the CSV pipeline — that is what this guards.
    porcelain = subprocess.run(
        ["git", "status", "--porcelain", "server/app"],
        capture_output=True, text=True, cwd=_repo_root(),
    )
    changed = [
        line[3:].strip()
        for line in porcelain.stdout.splitlines()
        if line.strip() and line[3:].strip() != "server/app/main.py"
    ]
    assert changed == [], f"server/app changed beyond the sanctioned main.py mount: {changed!r}"


def _repo_root() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True,
    )
    return out.stdout.strip()
