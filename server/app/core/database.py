import ssl
import urllib.parse
from collections.abc import Mapping
from typing import Any

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

_ASYNCPG_VALID_SSLMODE = frozenset(
    {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
)
_LOCAL_DB_HOSTS = frozenset({"", "localhost", "127.0.0.1", "::1"})
_SSL_QUERY_KEYS = frozenset({"sslmode", "ssl"})
_LIBPQ_ONLY_QUERY_KEYS = frozenset({"channel_binding"})


def _unverified_ssl_context() -> ssl.SSLContext:
    context = ssl.create_default_context()
    context.check_hostname = False
    context.verify_mode = ssl.CERT_NONE
    return context


def _is_local_database(parsed_url: urllib.parse.ParseResult) -> bool:
    return (parsed_url.hostname or "").lower() in _LOCAL_DB_HOSTS


def _normalise_sslmode(raw_value: str | None) -> str | None:
    if raw_value is None:
        return None

    value = raw_value.strip().lower()
    if value in {"true", "1", "yes", "on"}:
        return "require"
    if value in {"false", "0", "no", "off"}:
        return "disable"
    return value


def _asyncpg_ssl_arg(
    parsed_url: urllib.parse.ParseResult,
    sslmode: str | None,
) -> Any:
    normalised_sslmode = _normalise_sslmode(sslmode)

    if normalised_sslmode:
        if normalised_sslmode == "disable":
            return False
        if normalised_sslmode in _ASYNCPG_VALID_SSLMODE:
            return normalised_sslmode
        return _unverified_ssl_context()

    if _is_local_database(parsed_url):
        return False

    return "require"


def _sanitize_db_url(raw_url: str) -> tuple[str, Mapping[str, Any]]:
    """Remove sslmode from the DSN and pass SSL config via asyncpg args."""
    parsed = urllib.parse.urlparse(raw_url)
    query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    sslmode_values = [
        value
        for key, value in query_pairs
        if key.strip().lower() in _SSL_QUERY_KEYS
    ]
    remaining_pairs = [
        (key, value)
        for key, value in query_pairs
        if key.strip().lower() not in _SSL_QUERY_KEYS | _LIBPQ_ONLY_QUERY_KEYS
    ]

    new_query = urllib.parse.urlencode(remaining_pairs, doseq=True)
    clean_url = urllib.parse.urlunparse(parsed._replace(query=new_query))
    sslmode = sslmode_values[-1] if sslmode_values else None
    return clean_url, {"ssl": _asyncpg_ssl_arg(parsed, sslmode)}


_db_url, _ssl_extra = _sanitize_db_url(get_settings().DATABASE_URL)

engine = create_async_engine(
    _db_url,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,       # recycle connections every 5 min (Neon closes idle after ~5 min)
    # pool_size + max_overflow per worker process.
    # At 4 uvicorn workers: 4 × (20 + 30) = 200 max Postgres connections —
    # well within Azure Postgres Flexible Server's default 200-connection limit.
    # Old value (5 + 10  = 15 per worker) caused pool exhaustion under burst load.
    pool_size=20,
    max_overflow=30,
    connect_args={
        "server_settings": {"application_name": "danta-search"},
        "command_timeout": 30,
        **_ssl_extra,
    },
)
async_session = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    pass


async def get_db():
    async with async_session() as session:
        try:
            yield session
        finally:
            try:
                await session.close()
            except Exception:
                pass  # connection may already be closed (Neon idle timeout)
