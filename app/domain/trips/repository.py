"""Trip loading + serialization (no mutations).

This module owns:
  - `_serialize(trip)` — sync dict rendering of an already-loaded Trip
  - `serialize_trip(db, trip)` — async chokepoint that enriches (Google
    Places + Directions + static map URL) then serializes; every mutation
    path runs through this before returning
  - `_load_trip`, `_load_day`, `_load_stop` — auth-checked loaders with
    `selectinload` + `populate_existing=True` so cached enrichment state
    is fresh
  - `_touch_trip` — bumps `Trip.updated_at` on child mutations
  - `serialize_trip_for_context(trip)` — strips top-level metadata for
    system-prompt injection (refine mode)
  - `get_or_create_refine_conversation` — 1:1 lookup between a Trip and
    its refine conversation

Mutations live in `app/domain/trips/mutations.py` (which imports from
here).
"""
import logging
from uuid import UUID

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.models import Conversation, Trip, TripDay, TripStop, utcnow


logger = logging.getLogger(__name__)


def _serialize(trip: Trip) -> dict:
    return {
        "id": str(trip.id),
        "title": trip.title,
        "summary": trip.summary,
        "start_date": trip.start_date.isoformat() if trip.start_date else None,
        "static_map_url": trip.static_map_url,
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
                        "photo_url": stop.photo_url,
                        "photos": list(stop.photos or []),
                        "rating": stop.rating,
                        "formatted_address": stop.formatted_address,
                        "completed_at": (
                            stop.completed_at.isoformat()
                            if stop.completed_at else None
                        ),
                    }
                    for stop in day.stops
                ],
            }
            for day in trip.days
        ],
        "routes": [
            {
                "id": str(route.id),
                "trip_day_id": str(route.trip_day_id),
                "from_stop_id": str(route.from_stop_id),
                "to_stop_id": str(route.to_stop_id),
                "encoded_polyline": route.encoded_polyline,
                "distance_meters": route.distance_meters,
                "duration_seconds": route.duration_seconds,
                "travel_mode": route.travel_mode,
            }
            for route in (trip.routes or [])
        ],
    }


async def _enrich_and_serialize(db: AsyncSession, trip: Trip) -> dict:
    """Enrich the trip server-side, then serialize.

    The single chokepoint every trip-mutation path runs through — `POST
    /trips`, `PUT /trips/{id}`, every `apply_*` endpoint, and both
    `event: trip_refined` SSE branches in chat. Best-effort: enrichment
    failures are logged but the response still goes out (with whatever
    pre-enrichment state we had on disk).
    """
    try:
        # Local import dodges the repository ↔ enrichment cycle.
        from app.domain.trips.enrichment import enrich_trip
        await enrich_trip(db, trip.id)
        # enrich_trip mutated stop fields and replaced trip_routes in the
        # same session; reload so the serialized dict reflects fresh state.
        refreshed = await _load_trip(db, trip.id, trip.user_id)
        if refreshed is not None:
            trip = refreshed
    except Exception:
        logger.exception(
            "enrich_and_serialize trip_id=%s — falling back to raw serialize",
            trip.id,
        )
    return _serialize(trip)


# Public alias. routers/trips.py and routers/chat.py import this; the
# leading-underscore name stays callable too for any internal use.
serialize_trip = _enrich_and_serialize


async def _load_trip(
    db: AsyncSession, trip_id: UUID, user_id: int
) -> Trip | None:
    # populate_existing=True overwrites any stale state in the session
    # identity map. Required because the engine uses expire_on_commit=False:
    # when `enrich_trip` commits new trip_routes rows mid-request, the Trip
    # already in the identity map keeps its pre-enrichment `routes=[]`
    # snapshot, and selectinload alone won't refresh an already-loaded
    # collection. Without this, the API response serializes 0 routes even
    # when enrich_trip logged routes=N/N.
    result = await db.execute(
        select(Trip)
        .where(Trip.id == trip_id, Trip.user_id == user_id)
        .options(
            selectinload(Trip.days).selectinload(TripDay.stops),
            selectinload(Trip.routes),
        )
        .execution_options(populate_existing=True)
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
