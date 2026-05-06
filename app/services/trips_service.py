"""Trip mutations and serialization shared by the trips router and the chat
pipeline.

Lives outside `app.routers` so `app.routers.chat` can call into it without
importing a router module (which would invert the layering and cause circular
imports as the chat pipeline grows). The trips router re-exports `_serialize`
and `_load_trip` from here for source-compatibility with prior callers.
"""
from datetime import date
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Trip, TripDay, TripStop, utcnow


def _serialize(trip: Trip) -> dict:
    return {
        "id": str(trip.id),
        "title": trip.title,
        "summary": trip.summary,
        "start_date": trip.start_date.isoformat() if trip.start_date else None,
        "created_at": trip.created_at.isoformat() if trip.created_at else "",
        "updated_at": trip.updated_at.isoformat() if trip.updated_at else "",
        "days": [
            {
                "id": str(day.id),
                "day_number": day.day_number,
                "title": day.title,
                "base_city": day.base_city,
                "stops": [
                    {
                        "id": str(stop.id),
                        "sort_order": stop.sort_order,
                        "name": stop.name,
                        "kind": stop.kind,
                        "city": stop.city,
                        "address": stop.address,
                        "duration_minutes": stop.duration_minutes,
                        "arrival_time": stop.arrival_time,
                        "end_time": stop.end_time,
                        "transportation": stop.transportation,
                        "notes": stop.notes,
                        "lat": stop.lat,
                        "lng": stop.lng,
                        "google_place_id": stop.google_place_id,
                    }
                    for stop in day.stops
                ],
            }
            for day in trip.days
        ],
    }


async def _load_trip(
    db: AsyncSession, trip_id: UUID, user_id: int
) -> Trip | None:
    result = await db.execute(
        select(Trip)
        .where(Trip.id == trip_id, Trip.user_id == user_id)
        .options(selectinload(Trip.days).selectinload(TripDay.stops))
    )
    return result.unique().scalar_one_or_none()


async def replace_trip_from_payload(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    payload: dict,
) -> Trip:
    """Atomically replace a trip's days/stops from a validated TripCreate dict.

    Caller owns the transaction — this function flushes but does not commit, so
    a refine turn can land the trip replace and the assistant-message persist
    in one boundary. Raises HTTPException(404) if the trip is not owned by
    `user_id`.
    """
    trip = await _load_trip(db, trip_id, user_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")

    trip.title = payload["title"]
    trip.summary = payload.get("summary")
    sd = payload.get("start_date")
    trip.start_date = date.fromisoformat(sd) if sd else None
    trip.updated_at = utcnow()

    # Clear existing children — cascade="all, delete-orphan" wipes days then
    # stops as the assignment runs. Flush before re-adding so the
    # UNIQUE(trip_id, day_number) and UNIQUE(trip_day_id, sort_order)
    # constraints don't see overlapping old + new rows mid-transaction.
    trip.days = []
    await db.flush()

    for day in payload["days"]:
        new_day = TripDay(
            day_number=day["day_number"],
            title=day.get("title"),
            base_city=day.get("base_city"),
        )
        for idx, stop in enumerate(day["stops"]):
            new_day.stops.append(
                TripStop(
                    sort_order=idx,
                    name=stop["name"],
                    kind=stop["kind"],
                    city=stop.get("city"),
                    address=stop.get("address"),
                    duration_minutes=stop.get("duration_minutes"),
                    arrival_time=stop.get("arrival_time"),
                    end_time=stop.get("end_time"),
                    transportation=stop.get("transportation"),
                    notes=stop.get("notes"),
                    lat=stop.get("lat"),
                    lng=stop.get("lng"),
                    google_place_id=stop.get("google_place_id"),
                )
            )
        trip.days.append(new_day)
    await db.flush()

    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        # Should be impossible — we just flushed it.
        raise HTTPException(status_code=500, detail="Trip vanished after replace")
    return refreshed


def serialize_trip_for_context(trip: Trip) -> dict:
    """Render a Trip for system-prompt injection.

    Drops top-level metadata Claude can't use (id, created_at, updated_at) and
    keeps child IDs (day.id, stop.id) so the model can reason about specific
    stops in the conversation ("preserve stop X", "lengthen day 2's stop Y").
    """
    full = _serialize(trip)
    full.pop("id", None)
    full.pop("created_at", None)
    full.pop("updated_at", None)
    return full


async def get_or_create_refine_conversation(
    db: AsyncSession,
    *,
    trip: Trip,
    user_id: int,
) -> Conversation:
    """1:1 lookup: find or create the refine conversation for this trip.

    Caller commits. Idempotent — calling twice returns the same row.
    """
    result = await db.execute(
        select(Conversation).where(
            Conversation.trip_id == trip.id,
            Conversation.user_id == user_id,
        )
    )
    conv = result.scalar_one_or_none()
    if conv is not None:
        return conv

    conv = Conversation(
        user_id=user_id,
        trip_id=trip.id,
        title=(f"Refine: {trip.title}")[:200],
        messages=[],
    )
    db.add(conv)
    await db.flush()
    return conv
