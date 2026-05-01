from uuid import UUID

from pydantic import BaseModel, Field


MAX_MESSAGE_LENGTH = 4000
MAX_HISTORY_MESSAGES = 20


class ChatMessage(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=MAX_MESSAGE_LENGTH)


class ChatRequest(BaseModel):
    message: str = Field(max_length=MAX_MESSAGE_LENGTH)
    conversation_history: list[ChatMessage] = Field(
        default_factory=list, max_length=MAX_HISTORY_MESSAGES
    )
    images: list[str] | None = Field(default=None, max_length=5)
    # When provided alongside an authenticated user, server-side history
    # supersedes `conversation_history`. Anonymous requests ignore this field.
    conversation_id: UUID | None = None


class TripCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    itinerary: dict = Field(default_factory=dict)


class TripOut(BaseModel):
    id: str
    title: str
    itinerary: dict
    created_at: str
    updated_at: str


class UpdatePreferencesRequest(BaseModel):
    # All preference fields nullable — we never write hardcoded defaults
    # to the DB. The chat layer infers what's missing from conversation.
    budget: str | None = None
    travel_style: list[str] | None = None
    food: list[str] | None = None
    pace: str | None = None
    interests: list[str] | None = None
    group_type: str | None = None
    accommodation: str | None = None
    transport: str | None = None
    travel_persona: str | None = None
    # Safe default — never nullable.
    language: str = "en"
