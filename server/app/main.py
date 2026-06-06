import time
import logging as _logging
from contextlib import asynccontextmanager

# Suppress uvicorn's default access log — it prints raw URLs including OAuth
# codes and tokens in plaintext. Our log_requests middleware handles request
# logging in a structured, redacted format.
_logging.getLogger("uvicorn.access").handlers.clear()
_logging.getLogger("uvicorn.access").propagate = False

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text
from starlette.middleware.sessions import SessionMiddleware

from app.core.config import get_settings
from app.core.database import engine, Base, async_session
from app.core.logger import upload_logger, folder_logger, container_logger, auth_logger, chat_logger
from app.services.audit_log import record_request_audit
from app.core import metrics as _metrics
from app.api.v1.auth import router as auth_router
from app.api.v1.folders import router as folders_router
from app.api.v1.files import router as files_router
from app.api.v1.containers import router as containers_router
from app.api.v1.users import router as users_router
from app.api.v1.chat import router as chat_router
from app.api.v1.admin import router as admin_router
from app.api.v1.logs import router as logs_router
from app.api.v1.access import router as access_router
from app.api.v1.organizations import router as organizations_router
from app.api.v1.dashboards import router as dashboards_router
from app.api.v1.onboarding import router as onboarding_router
import app.models.file  # ensure File table is created
import app.models.access_request  # ensure AccessRequest table is created
import app.models.container  # ensure ContainerConfig table is created
import app.models.file_metadata  # ensure FileMetadata table is created
import app.models.file_analytics  # ensure FileAnalytics table is created
import app.models.column_key_registry  # ensure ColumnKeyRegistry table is created
import app.models.semantic_layer  # ensure semantic layer tables are created
import app.models.background_job  # ensure BackgroundJob table is created
import app.models.conversation  # ensure Conversation + Message tables are created
import app.models.organization  # ensure Organization table is created
import app.models.schema_dictionary  # ensure SchemaDictionary table is created
import app.models.server_log  # ensure ServerLog table is created
import app.models.dashboard  # ensure Dashboard + DashboardFolder tables are created
import app.models.erp_classification  # ensure ErpClassification table is created
import app.models.semantic_contract  # ensure SemanticContract table is created
import pdf_chat.models  # register pdf_chat ORM tables on the shared Base


async def _add_column_if_missing(conn, table: str, column: str, col_type: str) -> None:
    """Safely add a column to an existing table (no-op if it already exists)."""
    result = await conn.execute(
        text(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_name = :table AND column_name = :column"
        ),
        {"table": table, "column": column},
    )
    if not result.scalar():
        await conn.execute(text(f'ALTER TABLE "{table}" ADD COLUMN "{column}" {col_type}'))


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with engine.begin() as conn:
        # Create any brand-new tables
        await conn.run_sync(Base.metadata.create_all)

        # Migrate existing tables — add columns introduced after initial schema
        await _add_column_if_missing(conn, "files", "container_id", "VARCHAR(36) REFERENCES container_configs(id) ON DELETE CASCADE")
        await _add_column_if_missing(conn, "files", "blob_path", "VARCHAR(1000) UNIQUE")
        await _add_column_if_missing(conn, "files", "ingest_status", "VARCHAR(20) NOT NULL DEFAULT 'not_ingested'")
        await _add_column_if_missing(conn, "files", "uploaded_by_id", "VARCHAR(36) REFERENCES users(id) ON DELETE SET NULL")
        await _add_column_if_missing(conn, "files", "upload_duration_secs", "DOUBLE PRECISION")
        await _add_column_if_missing(conn, "files", "is_preprocessed", "BOOLEAN NOT NULL DEFAULT FALSE")
        await _add_column_if_missing(conn, "folders", "container_id", "VARCHAR(36) REFERENCES container_configs(id) ON DELETE CASCADE")
        await _add_column_if_missing(conn, "users", "is_admin", "BOOLEAN NOT NULL DEFAULT FALSE")
        await _add_column_if_missing(conn, "users", "role", "VARCHAR(20) NOT NULL DEFAULT 'user'")
        # Multi-tenancy: add organization_id FK to users (Phase 16)
        await _add_column_if_missing(
            conn,
            "users",
            "organization_id",
            "VARCHAR(36) REFERENCES organizations(id) ON DELETE SET NULL",
        )
        # Backfill: existing admins get role='admin'
        await conn.execute(text("UPDATE users SET role = 'admin' WHERE is_admin = TRUE AND role = 'user'"))
        # Backfill: file_metadata.container_id from files.container_id where it is NULL but files has it set
        await conn.execute(text("""
            UPDATE file_metadata fm
            SET container_id = f.container_id
            FROM files f
            WHERE fm.file_id = f.id
              AND fm.container_id IS NULL
              AND f.container_id IS NOT NULL
        """))

    # Retrieval-engine schema (pgvector + pg_trgm + new file_metadata columns)
    from app.migrations.retrieval_schema_upgrade import migrate as _retrieval_migrate
    try:
        await _retrieval_migrate()
    except Exception as exc:
        chat_logger.warning("retrieval_migration_failed", error=str(exc)[:300])

    # Domain access control schema (domain_tag on folders, allowed_domains on users)
    from app.migrations.domain_schema_upgrade import migrate as _domain_migrate
    try:
        await _domain_migrate()
    except Exception as exc:
        chat_logger.warning("domain_migration_failed", error=str(exc)[:300])

    # Schema-dictionary table: nullable parquet path + new source_blob_path
    from app.migrations.schema_dictionary_upgrade import migrate as _schema_dict_migrate
    try:
        await _schema_dict_migrate()
    except Exception as exc:
        chat_logger.warning("schema_dict_migration_failed", error=str(exc)[:300])

    # Cleaning config + quarantine audit columns
    from app.migrations.cleaning_config_upgrade import migrate as _cleaning_migrate
    try:
        await _cleaning_migrate()
    except Exception as exc:
        chat_logger.warning("cleaning_config_migration_failed", error=str(exc)[:300])

    # Ontology layer — column_semantic_roles + GIN index + relationship semantic_role
    from app.migrations.ontology_schema_upgrade import migrate as _ontology_migrate
    try:
        await _ontology_migrate()
    except Exception as exc:
        chat_logger.warning("ontology_migration_failed", error=str(exc)[:300])

    # Per-container semantic role extensions
    from app.migrations.semantic_config_upgrade import migrate as _semantic_config_migrate
    try:
        await _semantic_config_migrate()
    except Exception as exc:
        chat_logger.warning("semantic_config_migration_failed", error=str(exc)[:300])

    # Relationship fingerprint index — tenant-scoped database-backed hashmap
    from app.migrations.relationship_index_upgrade import migrate as _relationship_index_migrate
    try:
        await _relationship_index_migrate()
    except Exception as exc:
        chat_logger.warning("relationship_index_migration_failed", error=str(exc)[:300])

    # Semantic layer — entities, cardinality, approved/candidate business joins
    from app.migrations.semantic_layer_upgrade import migrate as _semantic_layer_migrate
    try:
        await _semantic_layer_migrate()
    except Exception as exc:
        chat_logger.warning("semantic_layer_migration_failed", error=str(exc)[:300])

    # Dashboard layer: dashboards + dashboard_folders (metadata-driven dashboards)
    from app.migrations.dashboard_upgrade import migrate as _dashboard_migrate
    try:
        await _dashboard_migrate()
    except Exception as exc:
        chat_logger.warning("dashboard_migration_failed", error=str(exc)[:300])

    # ERP business-context layer — per-file classification (source system,
    # module, polarity, process role). Powers GATE A + the semantic contract.
    from app.migrations.erp_classification_upgrade import migrate as _erp_classification_migrate
    try:
        await _erp_classification_migrate()
    except Exception as exc:
        chat_logger.warning("erp_classification_migration_failed", error=str(exc)[:300])

    # Danta Semantic Contract — the governed per-container surface the planner
    # and dry-plan gate reason against (declared joins, exposed columns).
    from app.migrations.semantic_contract_upgrade import migrate as _semantic_contract_migrate
    try:
        await _semantic_contract_migrate()
    except Exception as exc:
        chat_logger.warning("semantic_contract_migration_failed", error=str(exc)[:300])

    # Multi-tenant Org/RBAC overhaul — additive, idempotent. Order matters:
    # multi_container (3) backfills container.organization_id; folder_scope (4)
    # backfills folder.organization_id FROM that, so 3 must precede 4.
    for _mod_name, _label in [
        ("org_rbac_users_upgrade", "org_rbac_users"),
        ("org_rbac_org_upgrade", "org_rbac_org"),
        ("org_multi_container_upgrade", "org_multi_container"),
        ("org_folder_scope_upgrade", "org_folder_scope"),
        ("manager_domain_assignment_upgrade", "manager_domain_assignment"),
        ("org_ai_settings_upgrade", "org_ai_settings"),
        ("platform_admin_grant_upgrade", "platform_admin_grant"),
        ("local_auth_upgrade", "local_auth"),
        ("access_org_ai_upgrade", "access_org_ai"),
    ]:
        try:
            _mod = __import__(f"app.migrations.{_mod_name}", fromlist=["migrate"])
            await _mod.migrate()
        except Exception as exc:
            chat_logger.warning("org_rbac_migration_failed", migration=_label, error=str(exc)[:300])

    # Drop legacy audit_logs table — all audit events now go to server_logs
    from app.migrations.drop_audit_logs import migrate as _drop_audit_logs
    try:
        await _drop_audit_logs()
    except Exception as exc:
        chat_logger.warning("drop_audit_logs_failed", error=str(exc)[:300])

    # Phase 5: Ingestion trustworthiness columns on file_metadata + file_relationships
    from app.migrations.ingestion_trust_upgrade import run_upgrade as _ingestion_trust_upgrade
    try:
        async with engine.begin() as _conn:
            from sqlalchemy.ext.asyncio import AsyncSession as _AS
            async with _AS(_conn) as _sess:
                await _ingestion_trust_upgrade(_sess)
    except Exception as exc:
        chat_logger.warning("ingestion_trust_migration_failed", error=str(exc)[:300])

    # PDF chat module — additive runtime migrations (per-migration guard: one
    # failure must not block the others; same convention as org_rbac above).
    from pdf_chat.migrations.tunables_upgrade import (
        run_migration as _pdf_tunables_migration,
        install_db_lookup as _pdf_install_db_lookup,
    )
    from pdf_chat.migrations.bridge_upgrade import run_migration as _pdf_bridge_migration
    from pdf_chat.migrations.comprehension_upgrade import (
        apply_comprehension_migration as _pdf_comprehension_migration,
    )
    from pdf_chat.migrations.control_plane_upgrade import (
        run_migration as _pdf_control_plane_migration,
    )

    try:
        await _pdf_tunables_migration(engine)
        _pdf_install_db_lookup(async_session)
    except Exception as exc:
        chat_logger.warning("pdf_tunables_migration_failed", error=str(exc)[:300])
    try:
        await _pdf_bridge_migration(engine)
    except Exception as exc:
        chat_logger.warning("pdf_bridge_migration_failed", error=str(exc)[:300])
    try:
        await _pdf_comprehension_migration(engine)
    except Exception as exc:
        chat_logger.warning("pdf_comprehension_migration_failed", error=str(exc)[:300])
    try:
        await _pdf_control_plane_migration(engine)
    except Exception as exc:
        chat_logger.warning("pdf_control_plane_migration_failed", error=str(exc)[:300])

    # Pre-warm DataFusion  session pool — pays UDF-registration cost once at startup
    # so the first   N concurrent queries borrow a ready context without overhead.
    try:
        import asyncio as _asyncio
        from app.core.datafusion_client import warm_context_pool as _warm_pool
        await _asyncio.get_event_loop().run_in_executor(None, _warm_pool)
        chat_logger.info("datafusion_pool_startup", status="warmed")
    except Exception as exc:
        chat_logger.warning("datafusion_pool_startup_failed", error=str(exc)[:200])

    yield
    await engine.dispose()


settings = get_settings()

app = FastAPI(title="danta-search API", lifespan=lifespan)

# Session middleware required by authlib OAuth
app.add_middleware(SessionMiddleware, secret_key=settings.SECRET_KEY)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.FRONTEND_URL, "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.core.onboarding_gate import onboarding_gate_middleware

# HARD ONBOARDING GATE — blocks un-onboarded org_owners from all /api routes
# except onboarding/auth/health. Registered after log_requests so log_requests
# remains the outermost middleware and still records the 403.
app.middleware("http")(onboarding_gate_middleware)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    if request.method == "OPTIONS":
        return await call_next(request)

    start = time.perf_counter()
    response = None
    error: str | None = None
    try:
        response = await call_next(request)
    except Exception as exc:
        error = str(exc)[:500]
        raise
    finally:
        duration_ms = round((time.perf_counter() - start) * 1000, 2)
        status_code = response.status_code if response is not None else 500

        path = request.url.path
        method = request.method

        if "/files" in path:
            upload_logger.info("request", method=method, path=path, status=status_code, duration_ms=duration_ms)
        elif "/folders" in path:
            folder_logger.info("request", method=method, path=path, status=status_code, duration_ms=duration_ms)
        elif "/containers" in path:
            container_logger.info("request", method=method, path=path, status=status_code, duration_ms=duration_ms)
        elif "/auth" in path:
            auth_logger.info("request", method=method, path=path, status=status_code, duration_ms=duration_ms)

        await record_request_audit(
            request,
            status_code=status_code,
            duration_ms=duration_ms,
            error=error,
        )

    return response


app.include_router(auth_router, prefix="/api")
app.include_router(folders_router, prefix="/api")
app.include_router(files_router, prefix="/api")
app.include_router(containers_router, prefix="/api")
app.include_router(users_router, prefix="/api")
app.include_router(chat_router, prefix="/api")
app.include_router(admin_router, prefix="/api")
app.include_router(logs_router, prefix="/api")
app.include_router(access_router, prefix="/api")
app.include_router(organizations_router, prefix="/api")
app.include_router(dashboards_router, prefix="/api")
app.include_router(onboarding_router, prefix="/api")

# PDF chat module routers (self-prefixed: pdf_router=/api/pdf,
# pdf_onboarding_router=/api/pdf/onboarding). Aliased to avoid the name
# collision with the app's org onboarding_router above. Both routers derive the
# principal from pdf_chat.api.routes._resolve_current_user; binding it once to
# the app's get_current_user authenticates BOTH (onboarding reuses the same seam).
from pdf_chat.api.routes import pdf_router, _resolve_current_user as _pdf_resolve_user
from pdf_chat.api.onboarding import onboarding_router as pdf_onboarding_router
from app.dependencies import get_current_user as _app_get_current_user

app.dependency_overrides[_pdf_resolve_user] = _app_get_current_user
app.include_router(pdf_router)            # already prefixed with /api/pdf
app.include_router(pdf_onboarding_router) # already prefixed with /api/pdf/onboarding


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/metrics")
async def metrics():
    """Live in-process metrics snapshot.

    Returns query latency percentiles, error counts, blob bytes, and other
    key operational counters defined in RND_IMPLEMENTATION_PLAN Phase 4.
    No authentication required — restrict at the infra/network layer if needed.
    """
    return _metrics.get_snapshot()
