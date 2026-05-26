import ssl
import urllib.parse

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

# asyncpg only accepts these sslmode values; anything else causes a
# ClientConfigurationError at startup before the app can serve requests.
_ASYNCPG_VALID_SSLMODE = frozenset(
    {"disable", "allow", "prefer", "require", "verify-ca", "verify-full"}
)


def _sanitize_db_url(raw_url: str) -> tuple[str, dict]:
    """Strip or normalise a non-asyncpg sslmode so the engine never crashes.

    Returns (clean_url, extra_connect_args).  When the sslmode is invalid
    we remove it from the URL and pass an ssl context via connect_args so
    the connection is still encrypted (matching the intent of any non-empty
    sslmode value).
    """
    parsed = urllib.parse.urlparse(raw_url)
    params = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)

    sslmode_values = params.pop("sslmode", None)
    sslmode = sslmode_values[0] if sslmode_values else None

    extra: dict = {}

    if sslmode is None:
        # No sslmode at all — leave URL unchanged.
        return raw_url, extra

    if sslmode in _ASYNCPG_VALID_SSLMODE:
        # Valid — put it back unchanged.
        params["sslmode"] = [sslmode]
    else:
        # Non-standard value (e.g. "no-verify", "noverify", "true", …).
        # Build an SSL context that requires encryption but skips cert
        # verification, which matches the intent of such values.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        extra["ssl"] = ctx

    new_query = urllib.parse.urlencode(
        {k: v[0] for k, v in params.items()},
        quote_via=urllib.parse.quote,
    )
    clean_url = urllib.parse.urlunparse(parsed._replace(query=new_query))
    return clean_url, extra


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
