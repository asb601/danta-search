from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine, AsyncSession
from sqlalchemy.orm import DeclarativeBase
from app.core.config import get_settings

engine = create_async_engine(
    get_settings().DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_recycle=300,       # recycle connections every 5 min (Neon closes idle after ~5 min)
    # pool_size + max_overflow per worker process.
    # At 4 uvicorn workers: 4 × (20 + 30) = 200 max Postgres connections —
    # well within Azure Postgres Flexible Server's default 200-connection limit.
    # Old value (5 + 10 = 15 per worker) caused pool exhaustion under burst load.
    pool_size=20,
    max_overflow=30,
    connect_args={
        "server_settings": {"application_name": "danta-search"},
        "command_timeout": 30,
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
