from pathlib import Path
from pydantic_settings import BaseSettings
from functools import lru_cache


_INGESTION_POLICY_PROXY_NAMES = frozenset({
    "REINGEST_BATCH_SIZE",
    "REINGEST_BATCH_DELAY_SECONDS",
    "PARQUET_CONVERSION_CONCURRENCY",
    "CELERY_WORKER_CONCURRENCY",
    "CELERY_WORKER_PREFETCH_MULTIPLIER",
    "CELERY_RESULT_EXPIRES_SECONDS",
})


class Settings(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/genchatbot"

    # Google OAuth
    GOOGLE_CLIENT_ID: str = ""
    GOOGLE_CLIENT_SECRET: str = ""

    # JWT
    SECRET_KEY: str = "change-me-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days

    # Admin
    ADMIN_EMAIL: str = ""

    # Azure OpenAI
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_KEY: str = ""
    # Primary deployment — gpt-4o, used on turn 1
    AZURE_OPENAI_DEPLOYMENT: str = ""
    # Mini deployment — gpt-4o-mini, used on follow-up turns 2+
    AZURE_OPENAI_DEPLOYMENT_MINI: str = ""
    # Embedding model deployment
    AZURE_OPENAI_EMBEDDING_DEPLOYMENT: str = ""

    # Cost control: when true, every gpt-4o ("primary"/"standard"/"high") chat
    # call is routed to the gpt-4o-mini deployment instead — the heavy model is
    # never used. Embeddings (text-embedding-3-small) are unaffected.
    # Set DISABLE_GPT4O=false in .env to restore gpt-4o.
    DISABLE_GPT4O: bool = True

    # .env aliases (AZURE_OPENAI_API_BASE / AZURE_OPENAI_API_KEY)
    AZURE_OPENAI_API_BASE: str = ""
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_MODEL: str = ""  # legacy, ignored
    AZURE_OPENAI_API_VERSION: str = ""

    # SMTP (Gmail App Password recommended)
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASSWORD: str = ""

    # Query engine — "duckdb" (default, safe) | "datafusion" (new, concurrent)
    # Switch to "datafusion" once shadow testing confirms correctness.
    QUERY_ENGINE: str = "duckdb"

    # SQL validator AST mode — runtime override for the sqlglot structural validator.
    # "primary"  — AST is authoritative; regex runs as shadow (default, production).
    # "shadow"   — AST runs for telemetry only; regex remains authoritative.
    #              Use this to roll back to regex during an incident.
    # "disabled" — AST completely bypassed; regex only.
    SQL_VALIDATOR_AST_MODE: str = "primary"

    # Ingestion behavior is policy, not core application settings. The default
    # policy is external JSON, and production can replace it with a deployment
    # file or INGESTION_POLICY_JSON. Individual legacy env vars still override
    # the policy while the ingestion modules are migrated off direct Settings access.
    INGESTION_POLICY_FILE: str = "config/ingestion_policy.json"
    INGESTION_POLICY_JSON: str = ""

    # ── Org-RBAC v2 rollout flags (Lane B) ──────────────────────────────────
    # All default to a backward-compatible posture: with these defaults,
    # runtime behavior is byte-identical to the pre-overhaul system.
    #
    # RBAC_V2_ENFORCE — when True, resolve_chat_scope applies the new org+domain
    #   scoping via org_access. Default False keeps TODAY'S scoping exactly.
    # RBAC_V2_SHADOW  — when True (and ENFORCE False), additionally log what the
    #   new org+domain scope WOULD be, without changing behavior.
    # ORG_AI_KEYS_ENABLED — gate for per-org AI key resolution (OrgAISettings).
    # ONBOARDING_REQUIRED — gate for enforcing org onboarding completion.
    RBAC_V2_ENFORCE: bool = True
    RBAC_V2_SHADOW: bool = False
    ORG_AI_KEYS_ENABLED: bool = True
    ONBOARDING_REQUIRED: bool = True

    # ORG_LIVE_DB_ENABLED — gate for the live read-only org Postgres data source.
    #   Naturally gated: the org_postgres tools only activate when an org actually
    #   has a non-empty postgres_url resolved from OrgAISettings. When True, a
    #   resolved DSN registers two read-only LangChain tools (list_org_database,
    #   run_org_sql) alongside the Parquet tools; failures never break normal chat.
    # ORG_DB_QUERY_TIMEOUT_SECONDS — statement/connection timeout for live queries.
    # ORG_DB_MAX_ROWS — hard cap on rows returned by a live read-only SELECT.
    ORG_LIVE_DB_ENABLED: bool = True
    ORG_DB_QUERY_TIMEOUT_SECONDS: int = 15
    ORG_DB_MAX_ROWS: int = 1000

    # CORS
    FRONTEND_URL: str = "http://localhost:3000"

    # Storage encryption key (Fernet) — protects Azure connection strings at rest
    # Generate with: python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
    STORAGE_ENCRYPTION_KEY: str = ""

    # Redis — broker + result backend for Celery workers
    # Separate DBs: db=0 broker, db=1 results, db=2 response cache (future)
    # Azure Cache for Redis: use rediss:// (TLS) with the primary key as password
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_URL_RESULTS: str = "redis://localhost:6379/1"

    # OpenSearch — production metadata retrieval (BM25 + fuzzy + vector)
    # Empty URL disables OpenSearch and keeps PostgreSQL retrieval fallback active.
    OPENSEARCH_URL: str = ""
    OPENSEARCH_USERNAME: str = ""
    OPENSEARCH_PASSWORD: str = ""
    OPENSEARCH_API_KEY: str = ""
    OPENSEARCH_INDEX_PREFIX: str = "gchat-files"
    OPENSEARCH_TIMEOUT_SECONDS: float = 5.0
    OPENSEARCH_SHARDS: int = 1
    OPENSEARCH_REPLICAS: int = 0

    model_config = {"env_file": str(Path(__file__).resolve().parent.parent.parent / ".env"), "extra": "ignore"}

    def chat_deployment(self) -> str:
        """Resolve the deployment for the primary/standard/high chat lane.

        Cost control: when DISABLE_GPT4O is set, returns the gpt-4o-mini
        deployment so the heavy model is never selected. Falls back to the
        primary deployment if mini is not configured (so a misconfigured env
        does not break chat). Preserves the legacy 'gpt-4'→model alias.
        """
        if self.DISABLE_GPT4O and self.AZURE_OPENAI_DEPLOYMENT_MINI:
            return self.AZURE_OPENAI_DEPLOYMENT_MINI
        primary = self.AZURE_OPENAI_DEPLOYMENT
        if primary == "gpt-4" and self.AZURE_OPENAI_MODEL:
            return self.AZURE_OPENAI_MODEL
        return primary

    def __getattr__(self, name: str):
        if name.startswith("INGEST_") or name in _INGESTION_POLICY_PROXY_NAMES:
            from app.services.ingestion_policy import get_ingestion_policy

            return get_ingestion_policy().legacy_value(name)
        return super().__getattr__(name)


@lru_cache
def get_settings() -> Settings:
    return Settings()
