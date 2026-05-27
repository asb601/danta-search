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

    # Governed SemanticMemory / BrainContext runtime caps. These are deployment
    # tunables; they do not encode tenant-specific business logic.
    BRAIN_CONTEXT_MAX_RECORDS: int = 8
    BRAIN_CONTEXT_MAX_DOMAINS: int = 6
    BRAIN_CONTEXT_MAX_CANDIDATES: int = 60
    BRAIN_CONTEXT_MAX_TERMS: int = 24
    BRAIN_CONTEXT_MAX_ANCHOR_FILES: int = 8
    BRAIN_CONTEXT_TOKEN_BUDGET: int = 900
    BRAIN_CONTEXT_MIN_SCORE: float = 0.12
    BRAIN_CONTEXT_TRACE_ENABLED: bool = True
    PLAN_IR_MAX_STAGES: int = 6
    PLAN_IR_MAX_CONTRACTS: int = 12
    SEMANTIC_DOMAIN_MAX_SOURCE_MEMORIES: int = 800
    SEMANTIC_DOMAIN_MAX_CLUSTERS: int = 120
    SEMANTIC_DOMAIN_MIN_FILES: int = 2
    SEMANTIC_DOMAIN_MAX_TERMS: int = 28
    SEMANTIC_DOMAIN_DECAY_PER_CONFLICT: float = 0.08

    model_config = {"env_file": str(Path(__file__).resolve().parent.parent.parent / ".env"), "extra": "ignore"}

    def __getattr__(self, name: str):
        if name.startswith("INGEST_") or name in _INGESTION_POLICY_PROXY_NAMES:
            from app.services.ingestion_policy import get_ingestion_policy

            return get_ingestion_policy().legacy_value(name)
        return super().__getattr__(name)


@lru_cache
def get_settings() -> Settings:
    return Settings()
