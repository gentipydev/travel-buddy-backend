from datetime import datetime, timezone
import uuid

from sqlalchemy import Column, String, Text, Integer, Float, ForeignKey, DateTime, Date, Index, CheckConstraint, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB, UUID, ARRAY, TSVECTOR
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector


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
    # Nullable — never assume; defaults should emerge from conversation,
    # not from the schema. Wrong defaults are worse than no defaults.
    budget = Column(String, nullable=True, default=None)
    travel_style = Column(JSONB, nullable=True, default=None)
    food = Column(JSONB, nullable=True, default=None)
    pace = Column(String, nullable=True, default=None)
    interests = Column(JSONB, nullable=True, default=None)
    group_type = Column(String, nullable=True, default=None)
    accommodation = Column(String, nullable=True, default=None)
    transport = Column(String, nullable=True, default=None)
    travel_persona = Column(Text, nullable=True, default=None)
    # Safe default — language is never nullable.
    language = Column(String, nullable=False, default="en")
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="preferences")

    def to_prompt(self) -> str:
        """Render only the fields the user has actually set. Every preference
        is labeled as a *default* — the override reminder in
        `build_system_prompt` tells Claude that conversation always wins.
        """
        parts: list[str] = []
        if self.travel_persona:
            parts.append(
                f"Travel persona (default vibe — defer to conversation): "
                f"{self.travel_persona}"
            )
        if self.budget:
            parts.append(f"- Default budget: {self.budget}")
        if self.group_type:
            parts.append(f"- Default group type: {self.group_type}")
        if self.travel_style:
            parts.append(f"- Default travel style: {', '.join(self.travel_style)}")
        if self.accommodation:
            parts.append(f"- Default accommodation: {self.accommodation}")
        if self.transport:
            parts.append(f"- Default transport: {self.transport}")
        if self.pace:
            parts.append(f"- Default pace: {self.pace}")
        if self.food:
            parts.append(f"- Food preferences: {', '.join(self.food)}")
        if self.interests:
            parts.append(f"- Interests: {', '.join(self.interests)}")
        parts.append(f"- Preferred language: {self.language}")
        return "\n".join(parts)

    def has_preferences(self) -> bool:
        """True if at least one meaningful preference has been set."""
        return any([
            self.budget,
            self.travel_style,
            self.food,
            self.pace,
            self.interests,
            self.group_type,
            self.accommodation,
            self.transport,
            self.travel_persona,
        ])


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
    # 1:1 link to a saved trip when this conversation was opened via "Refine
    # with Buddy". NULL for regular chat. Partial unique index lives in
    # migration 012; ON DELETE SET NULL preserves history if the trip is
    # deleted.
    trip_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trips.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    title = Column(String, nullable=True)
    messages = Column(JSONB, nullable=False, default=list)
    summary = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    trip = relationship("Trip", back_populates="conversation", uselist=False)


class Trip(Base):
    """Normalized itinerary owned by a user. Children: TripDay → TripStop.

    Pre-launch the prior `place_id`/`itinerary JSONB` columns were dropped
    in migration 011; nothing in the codebase references them anymore.
    """
    __tablename__ = "trips"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = Column(String, nullable=False)
    summary = Column(Text, nullable=True)
    start_date = Column(Date, nullable=True)
    created_at = Column(DateTime(timezone=True), default=utcnow)
    updated_at = Column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)

    user = relationship("User", back_populates="trips")
    days = relationship(
        "TripDay",
        back_populates="trip",
        cascade="all, delete-orphan",
        order_by="TripDay.day_number",
    )
    # Refine-with-Buddy conversation, if one was created. Do NOT cascade
    # delete — the FK on Conversation.trip_id is ON DELETE SET NULL, so the
    # chat history survives a trip deletion.
    conversation = relationship(
        "Conversation",
        back_populates="trip",
        uselist=False,
    )


class TripDay(Base):
    __tablename__ = "trip_days"
    __table_args__ = (
        UniqueConstraint("trip_id", "day_number", name="trip_days_trip_day_uniq"),
        CheckConstraint("day_number >= 1", name="trip_days_day_number_positive"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trip_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trips.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    day_number = Column(Integer, nullable=False)
    title = Column(String, nullable=True)
    base_city = Column(String, nullable=True)

    trip = relationship("Trip", back_populates="days")
    stops = relationship(
        "TripStop",
        back_populates="day",
        cascade="all, delete-orphan",
        order_by="TripStop.sort_order",
    )


class TripStop(Base):
    __tablename__ = "trip_stops"
    __table_args__ = (
        UniqueConstraint("trip_day_id", "sort_order", name="trip_stops_day_order_uniq"),
    )

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    trip_day_id = Column(
        UUID(as_uuid=True),
        ForeignKey("trip_days.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sort_order = Column(Integer, nullable=False)
    name = Column(String, nullable=False)
    kind = Column(String, nullable=False)
    city = Column(String, nullable=True)
    address = Column(Text, nullable=True)
    duration_minutes = Column(Integer, nullable=True)
    arrival_time = Column(String, nullable=True)
    end_time = Column(String, nullable=True)
    transportation = Column(String, nullable=True)
    notes = Column(Text, nullable=True)
    lat = Column(Float, nullable=True)
    lng = Column(Float, nullable=True)
    google_place_id = Column(String, nullable=True)

    day = relationship("TripDay", back_populates="stops")


Index(
    "chunks_embedding_idx",
    Chunk.embedding,
    postgresql_using="hnsw",
    postgresql_with={"m": 16, "ef_construction": 64},
    postgresql_ops={"embedding": "vector_cosine_ops"},
)


class Artifact(Base):
    """Curated Albanian artifact tied to one of the 8 mobile-app destinations.

    Image bytes live in the `artifact-photos` Supabase Storage bucket at
    `image_path`; the manifest endpoints return the public URL. `category`
    distinguishes featured artifacts ('highlight') from the smaller hidden
    set surfaced through the collectibles UI ('collectible'). `search_text`
    is a generated TSVECTOR populated by a trigger (see migration 010).
    """
    __tablename__ = "artifacts"

    id = Column(String, primary_key=True)
    destination_slug = Column(String, nullable=False, index=True)
    category = Column(String, nullable=False)
    sort_order = Column(Integer, nullable=False, default=0)
    title = Column(String, nullable=False)
    date_text = Column(String)
    period = Column(String)
    country = Column(String)
    culture = Column(String)
    medium = Column(String)
    dimension = Column(String)
    classification = Column(String)
    object_type = Column(String)
    object_begin_year = Column(Integer)
    object_end_year = Column(Integer)
    image_path = Column(String, nullable=False)
    icon_name = Column(String)
    search_text = Column(TSVECTOR)


Index(
    "artifacts_destination_idx",
    Artifact.destination_slug,
    Artifact.category,
    Artifact.sort_order,
)
Index(
    "artifacts_search_idx",
    Artifact.search_text,
    postgresql_using="gin",
)
