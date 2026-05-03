from typing import AsyncGenerator
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker

from app.config import settings
from app.db.models import User


# Supabase transaction pooler (port 6543) is pgbouncer in transaction mode.
# Three independent caches must be defeated for this to work reliably:
#   1. asyncpg's internal cache (statement_cache_size)         — connect_args
#   2. SQLAlchemy's PreparedStatement cache (prepared_statement_cache_size)
#   3. Statement-name collisions across worker processes
#      (prepared_statement_name_func with UUID-based names)
# Without #2, SQLAlchemy hands a stale PreparedStatement to a backend
# connection pgbouncer has since reassigned, raising
# InvalidSQLStatementNameError ("prepared statement ... does not exist").
engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex[:16]}__",
        # Cap initial connect / handshake so a half-dead connection can't
        # hang the pre_ping query indefinitely on Render free-tier wakeup.
        "command_timeout": 10,
    },
    pool_pre_ping=True,
    # Recycle connections every 30 min so silently-dropped TCP sockets on
    # Render's idle workers can't sit in the pool past their useful life.
    pool_recycle=1800,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)


async def init_db():
    """No-op: schema is managed via supabase/migrations/ run in the SQL Editor.
    pgvector is enabled there too."""
    return


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session() as session:
        yield session


async def get_or_create_user(db: AsyncSession, supabase_user: dict) -> User:
    """Find user by supabase_uid or create new one."""
    result = await db.execute(
        select(User).where(User.supabase_uid == supabase_user["id"])
    )
    user = result.scalar_one_or_none()
    if not user:
        email = supabase_user.get("email", "") or ""
        user = User(
            supabase_uid=supabase_user["id"],
            name=email.split("@")[0] if email else "user",
            email=email,
        )
        db.add(user)
        await db.commit()
        await db.refresh(user)
    return user
