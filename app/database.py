from datetime import datetime, timezone
from typing import AsyncGenerator
import uuid

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, DateTime, Index, select
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

from app.config import settings

# Supabase transaction pooler (port 6543) is pgbouncer in transaction mode —
# prepared statements break across connection swaps. Disable the cache AND
# randomize statement names: pgbouncer's backend-connection reuse causes
# DuplicatePreparedStatementError when asyncpg's default __asyncpg_stmt_N__
# names collide across processes.
engine = create_async_engine(
    settings.database_url,
    echo=False,
    connect_args={
        "statement_cache_size": 0,
        "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4().hex[:16]}__",
    },
    pool_pre_ping=True,
)
async_session = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

Base = declarative_base()


def utcnow():
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    supabase_uid = Column(String, unique=True, index=True, nullable=True)
    name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    preferences = relationship("UserPreferences", back_populates="user", uselist=False, cascade="all, delete-orphan")
    trips = relationship("Trip", back_populates="user", cascade="all, delete-orphan")


class UserPreferences(Base):
    __tablename__ = "user_preferences"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True, nullable=False, index=True)
    budget = Column(String, default="mid-range")
    travel_style = Column(JSONB, default=list)
    food = Column(JSONB, default=list)
    pace = Column(String, default="moderate")
    interests = Column(JSONB, default=list)
    language = Column(String, default="en")
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="preferences")

    def to_prompt(self) -> str:
        parts = [f"- Budget: {self.budget}"]
        if self.travel_style:
            parts.append(f"- Travel style: {', '.join(self.travel_style)}")
        if self.food:
            parts.append(f"- Food preferences: {', '.join(self.food)}")
        parts.append(f"- Pace: {self.pace}")
        if self.interests:
            parts.append(f"- Interests: {', '.join(self.interests)}")
        parts.append(f"- Preferred language: {self.language}")
        return "\n".join(parts)


class Place(Base):
    __tablename__ = "places"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    google_place_id = Column(String, unique=True, index=True)
    name = Column(String, nullable=False)
    city = Column(String, index=True)
    region = Column(String, index=True)
    category = Column(String, index=True)
    tags = Column(ARRAY(String))
    latitude = Column(Float)
    longitude = Column(Float)
    rating = Column(Float)
    price_level = Column(Integer)
    photos = Column(JSONB, default=list)  # public Supabase Storage URLs
    description = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    chunks = relationship("Chunk", back_populates="place", cascade="all, delete-orphan")


class Chunk(Base):
    __tablename__ = "chunks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    place_id = Column(UUID(as_uuid=True), ForeignKey("places.id", ondelete="CASCADE"), nullable=True, index=True)
    content = Column(String, nullable=False)
    language = Column(String, default="en", index=True)
    source = Column(String)
    embedding = Column(Vector(1024))  # cohere multilingual = 1024 dims
    meta = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)

    place = relationship("Place", back_populates="chunks")


class Conversation(Base):
    """Server-side chat session. `messages` is a JSONB array of
    {role, content, timestamp}. Once it grows past 20 entries the oldest 15
    are folded into `summary` (see services.conversation_service) so the LLM
    context stays bounded while long-term memory is preserved.

    Anonymous users do not get a Conversation row — `user_id` is nullable for
    schema flexibility, but the /chat/stream path only persists when the
    requester is authenticated AND has a synced local users row.

    Migration (also in supabase/migrations/006_create_conversations.sql):

        CREATE TABLE IF NOT EXISTS public.conversations (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            user_id INTEGER REFERENCES public.users(id) ON DELETE CASCADE,
            title TEXT,
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            summary TEXT,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS conversations_user_id_idx
            ON public.conversations (user_id);
        CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
            ON public.conversations (user_id, updated_at DESC);
    """
    __tablename__ = "conversations"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=True,
        index=True,
    )
    title = Column(String, nullable=True)
    messages = Column(JSONB, nullable=False, default=list)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)


class Trip(Base):
    __tablename__ = "trips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False)
    itinerary = Column(JSONB, default=dict)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="trips")


Index(
    "chunks_embedding_idx",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


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
