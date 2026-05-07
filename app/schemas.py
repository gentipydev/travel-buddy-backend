from datetime import date
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


StopKind = Literal[
    "sight", "beach", "hike", "meal", "stay", "transport", "activity"
]
Transportation = Literal["walk", "taxi", "tram", "bus", "car", "other"]


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
    # When set, this turn refines the named saved trip. The pipeline loads the
    # trip (auth-checked), injects its serialized state into the system prompt,
    # auto-resolves the 1:1 conversation if `conversation_id` is omitted, and
    # if Claude calls `emit_itinerary` will replace the trip in place rather
    # than emitting an unsaved itinerary card. Anonymous requests are rejected
    # with 404 (no local user, nothing to refine).
    trip_id: UUID | None = None


class TripStopCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    kind: StopKind
    city: str | None = None
    address: str | None = None
    duration_minutes: int | None = Field(default=None, ge=0)
    arrival_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    end_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    transportation: Transportation | None = None
    notes: str | None = None
    lat: float | None = None
    lng: float | None = None
    google_place_id: str | None = None


class TripDayCreate(BaseModel):
    day_number: int = Field(ge=1)
    title: str | None = None
    base_city: str | None = None
    stops: list[TripStopCreate] = Field(min_length=1)


class TripCreate(BaseModel):
    title: str = Field(min_length=1, max_length=200)
    summary: str | None = None
    start_date: date | None = None
    days: list[TripDayCreate] = Field(min_length=1)


class TripStopOut(TripStopCreate):
    id: str
    sort_order: int


class TripDayOut(TripDayCreate):
    id: str
    stops: list[TripStopOut]


class TripOut(BaseModel):
    id: str
    title: str
    summary: str | None = None
    start_date: date | None = None
    days: list[TripDayOut]
    created_at: str
    updated_at: str


class TripUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str | None = Field(default=None, min_length=1, max_length=200)
    summary: str | None = None
    start_date: date | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_null_title(cls, data):
        if isinstance(data, dict) and "title" in data and data["title"] is None:
            raise ValueError("title cannot be cleared")
        return data


class TripStopUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(default=None, min_length=1, max_length=200)
    kind: StopKind | None = None
    city: str | None = None
    address: str | None = None
    duration_minutes: int | None = Field(default=None, ge=0)
    arrival_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    end_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    transportation: Transportation | None = None
    notes: str | None = None
    lat: float | None = None
    lng: float | None = None
    google_place_id: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _reject_null_required(cls, data):
        if not isinstance(data, dict):
            return data
        for key in ("name", "kind"):
            if key in data and data[key] is None:
                raise ValueError(f"{key} cannot be cleared")
        return data


class TripStopCreateBody(TripStopCreate):
    """POST /trips/{id}/days/{did}/stops body. Same fields as TripStopCreate
    plus optional sort_order. Omit sort_order to append; provide it to insert
    at a position (existing stops at >= that position are shifted up by 1)."""
    sort_order: int | None = Field(default=None, ge=0)


class TripDayCreateBody(BaseModel):
    """POST /trips/{id}/days body. day_number is required; the client picks
    it (so renumbering is the client's responsibility, and bugs are loud).
    stops can be empty — decoupled from create-trip context which requires
    at least one stop per day."""
    model_config = ConfigDict(extra="forbid")
    day_number: int = Field(ge=1)
    title: str | None = None
    base_city: str | None = None
    stops: list[TripStopCreate] = Field(default_factory=list)


class StopReorderItem(BaseModel):
    stop_id: UUID
    sort_order: int = Field(ge=0)
    # Optional time slot fields. The reorder caller can include these per
    # item to anchor the chronological time slots to position (so swapping
    # two stops keeps `arrival_time` ascending). Only fields explicitly set
    # by the caller are forwarded — see the `exclude_unset=True` dump in the
    # router so a legacy {stop_id, sort_order} payload leaves times untouched.
    arrival_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    end_time: str | None = Field(
        default=None, pattern=r"^([01]\d|2[0-3]):[0-5]\d$"
    )
    duration_minutes: int | None = Field(default=None, ge=0)


class StopReorderBody(BaseModel):
    """PATCH /trips/{id}/days/{did}/reorder body. Caller sends the COMPLETE
    new ordering so the server can validate the set matches what's on disk."""
    model_config = ConfigDict(extra="forbid")
    order: list[StopReorderItem] = Field(min_length=1)


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
