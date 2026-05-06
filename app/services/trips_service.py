"""Trip mutations and serialization shared by the trips router and the chat
pipeline.

Lives outside `app.routers` so `app.routers.chat` can call into it without
importing a router module (which would invert the layering and cause circular
imports as the chat pipeline grows). The trips router re-exports `_serialize`
and `_load_trip` from here for source-compatibility with prior callers.

The `apply_*` functions are the no-commit primitives used by both the manual
editor endpoints and Claude's refine tools. Each loads + mutates + flushes,
returning the canonical post-mutation Trip; the caller owns the transaction
boundary so a refine turn can apply ops + persist the assistant message in
one commit.
"""
from datetime import date
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import case, select, update
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


async def _load_day(
    db: AsyncSession, trip_id: UUID, day_id: UUID, user_id: int
) -> TripDay | None:
    result = await db.execute(
        select(TripDay)
        .join(Trip, Trip.id == TripDay.trip_id)
        .where(
            TripDay.id == day_id,
            Trip.id == trip_id,
            Trip.user_id == user_id,
        )
        .options(selectinload(TripDay.stops))
    )
    return result.unique().scalar_one_or_none()


async def _load_stop(
    db: AsyncSession, trip_id: UUID, stop_id: UUID, user_id: int
) -> TripStop | None:
    result = await db.execute(
        select(TripStop)
        .join(TripDay, TripDay.id == TripStop.trip_day_id)
        .join(Trip, Trip.id == TripDay.trip_id)
        .where(
            TripStop.id == stop_id,
            Trip.id == trip_id,
            Trip.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def _touch_trip(db: AsyncSession, trip_id: UUID) -> None:
    """Bump Trip.updated_at on child mutations. SQLAlchemy's `onupdate=utcnow`
    only fires when the trip row itself is touched; for child mutations we
    have to update it explicitly in the same transaction."""
    await db.execute(
        update(Trip).where(Trip.id == trip_id).values(updated_at=utcnow())
    )


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


async def apply_update_trip(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    payload: dict,
) -> Trip:
    """Patch trip-level fields (title / summary / start_date). `payload` is the
    Pydantic `TripUpdate.model_dump(exclude_unset=True)` — only keys present
    are written. No-commit: caller owns the transaction."""
    trip = await _load_trip(db, trip_id, user_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    for key, value in payload.items():
        setattr(trip, key, value)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update")
    return refreshed


async def apply_add_day(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    payload: dict,
) -> Trip:
    """Insert a new day. `payload` matches `TripDayCreateBody.model_dump()`:
    `day_number` (required), optional `title` / `base_city`, and a `stops` list
    (may be empty). Raises HTTPException(409) on duplicate `day_number`."""
    trip = await _load_trip(db, trip_id, user_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    if any(d.day_number == payload["day_number"] for d in trip.days):
        raise HTTPException(
            status_code=409, detail="day_number already exists"
        )
    new_day = TripDay(
        trip_id=trip.id,
        day_number=payload["day_number"],
        title=payload.get("title"),
        base_city=payload.get("base_city"),
    )
    for index, stop in enumerate(payload.get("stops") or []):
        new_day.stops.append(
            TripStop(
                sort_order=index,
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
    db.add(new_day)
    await _touch_trip(db, trip.id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after add_day")
    return refreshed


async def apply_delete_day(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
) -> Trip:
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    await db.delete(day)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after delete_day")
    return refreshed


async def apply_insert_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
    payload: dict,
) -> Trip:
    """Insert a stop. `payload` matches `TripStopCreateBody.model_dump()`:
    all `TripStopCreate` fields plus optional `sort_order`. Omit `sort_order`
    to append; provide it to insert at a position (existing stops at >= that
    index are shifted up by 1 in a single UPDATE)."""
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    body = dict(payload)
    requested = body.pop("sort_order", None)
    if requested is None:
        next_sort = max((s.sort_order for s in day.stops), default=-1) + 1
        new_stop = TripStop(trip_day_id=day.id, sort_order=next_sort, **body)
    else:
        await db.execute(
            update(TripStop)
            .where(
                TripStop.trip_day_id == day.id,
                TripStop.sort_order >= requested,
            )
            .values(sort_order=TripStop.sort_order + 1)
        )
        new_stop = TripStop(trip_day_id=day.id, sort_order=requested, **body)
    db.add(new_stop)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after insert_stop")
    return refreshed


async def apply_update_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    stop_id: UUID,
    payload: dict,
) -> Trip:
    """Patch a single stop's fields. `payload` is
    `TripStopUpdate.model_dump(exclude_unset=True)` — only set keys are
    written, leaving everything else (including google_place_id, lat, lng,
    sort_order) untouched."""
    stop = await _load_stop(db, trip_id, stop_id, user_id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    for key, value in payload.items():
        setattr(stop, key, value)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update_stop")
    return refreshed


async def apply_delete_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    stop_id: UUID,
) -> Trip:
    stop = await _load_stop(db, trip_id, stop_id, user_id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    # Per design: do NOT compact remaining sort_orders on delete. Gaps are
    # fine; the relationship's `order_by="TripStop.sort_order"` handles it.
    await db.delete(stop)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after delete_stop")
    return refreshed


async def apply_reorder_stops(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
    payload: dict,
) -> Trip:
    """Reorder stops within a day. `payload` matches
    `StopReorderBody.model_dump()` with `order: list[{stop_id, sort_order}]`.
    Caller must send the COMPLETE new ordering so the server can validate the
    set matches what's on disk."""
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    order_items = payload.get("order") or []
    # Coerce stop_id strings → UUID for set comparison parity with ORM ids.
    parsed_order = [
        (UUID(str(item["stop_id"])), int(item["sort_order"]))
        for item in order_items
    ]
    current_ids = {s.id for s in day.stops}
    body_ids = {sid for sid, _ in parsed_order}
    if current_ids != body_ids:
        raise HTTPException(status_code=409, detail="stop set out of sync")
    body_orders = sorted(so for _, so in parsed_order)
    if body_orders != list(range(len(body_orders))):
        raise HTTPException(
            status_code=400,
            detail="sort_order values must be 0..N-1 contiguous",
        )
    new_order = {sid: so for sid, so in parsed_order}
    # Two-phase update to dodge mid-statement UNIQUE(trip_day_id, sort_order)
    # violations on a permutation. Despite the docs implying constraints are
    # checked at end-of-statement, PostgreSQL evaluates non-deferrable UNIQUE
    # per-row during UPDATE — so a swap (e.g. row at sort_order=1 → 2 while
    # row at sort_order=2 still sits at 2) fails the moment the first row
    # collides. Shifting every stop in the day into a disjoint negative range
    # first guarantees the positive 0..N-1 target range is free for phase 2.
    # Validation above ensures `current_ids == body_ids`, so every row in the
    # day is addressed by the CASE in phase 2 (no row gets NULL from a
    # missing WHEN branch).
    SHIFT = 1_000_000
    await db.execute(
        update(TripStop)
        .where(TripStop.trip_day_id == day.id)
        .values(sort_order=TripStop.sort_order - SHIFT)
    )
    await db.execute(
        update(TripStop)
        .where(TripStop.trip_day_id == day.id)
        .values(sort_order=case(new_order, value=TripStop.id))
    )
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after reorder")
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
