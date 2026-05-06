from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import case, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.engine import get_session
from app.db.models import Trip, TripDay, TripStop, User, utcnow
from app.deps import require_local_user
from app.schemas import (
    StopReorderBody,
    TripCreate,
    TripDayCreateBody,
    TripOut,
    TripStopCreateBody,
    TripStopUpdate,
    TripUpdate,
)
from app.services.trips_service import (
    _load_trip,
    _serialize,
    get_or_create_refine_conversation,
    replace_trip_from_payload,
)

# TODO(tests): pytest scaffolding deferred to a follow-up PR — see
# manual curl smoke test recipes in the trip editing plan.

router = APIRouter(prefix="/trips", tags=["trips"])


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


@router.post("", response_model=TripOut, status_code=status.HTTP_201_CREATED)
async def create_trip(
    payload: TripCreate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    trip = Trip(
        user_id=user.id,
        title=payload.title,
        summary=payload.summary,
        start_date=payload.start_date,
    )
    for day in payload.days:
        trip_day = TripDay(
            day_number=day.day_number,
            title=day.title,
            base_city=day.base_city,
        )
        for index, stop in enumerate(day.stops):
            trip_day.stops.append(
                TripStop(
                    sort_order=index,
                    name=stop.name,
                    kind=stop.kind,
                    city=stop.city,
                    address=stop.address,
                    duration_minutes=stop.duration_minutes,
                    arrival_time=stop.arrival_time,
                    end_time=stop.end_time,
                    transportation=stop.transportation,
                    notes=stop.notes,
                    lat=stop.lat,
                    lng=stop.lng,
                    google_place_id=stop.google_place_id,
                )
            )
        trip.days.append(trip_day)
    db.add(trip)
    await db.commit()

    refreshed = await _load_trip(db, trip.id, user.id)
    if refreshed is None:
        # Should be impossible — we just committed it. Defensive.
        raise HTTPException(status_code=500, detail="Trip not found after create")
    return _serialize(refreshed)


@router.get("", response_model=list[TripOut])
async def list_trips(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(Trip)
        .where(Trip.user_id == user.id)
        .options(selectinload(Trip.days).selectinload(TripDay.stops))
        .order_by(Trip.updated_at.desc())
    )
    trips = result.unique().scalars().all()
    return [_serialize(t) for t in trips]


@router.get("/{trip_id}", response_model=TripOut)
async def get_trip(
    trip_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    trip = await _load_trip(db, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    return _serialize(trip)


@router.delete("/{trip_id}")
async def delete_trip(
    trip_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(Trip).where(Trip.id == trip_id, Trip.user_id == user.id)
    )
    trip = result.scalar_one_or_none()
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    await db.delete(trip)
    await db.commit()
    return {"status": "deleted"}


@router.patch("/{trip_id}", response_model=TripOut)
async def update_trip(
    trip_id: UUID,
    payload: TripUpdate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    trip = await _load_trip(db, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(trip, key, value)
    # `onupdate=utcnow` on Trip itself fires on the attribute changes above.
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update")
    return _serialize(refreshed)


@router.post(
    "/{trip_id}/days",
    response_model=TripOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_day(
    trip_id: UUID,
    payload: TripDayCreateBody,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    trip = await _load_trip(db, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    if any(d.day_number == payload.day_number for d in trip.days):
        raise HTTPException(
            status_code=409, detail="day_number already exists"
        )
    new_day = TripDay(
        trip_id=trip.id,
        day_number=payload.day_number,
        title=payload.title,
        base_city=payload.base_city,
    )
    for index, stop in enumerate(payload.stops):
        new_day.stops.append(
            TripStop(
                sort_order=index,
                name=stop.name,
                kind=stop.kind,
                city=stop.city,
                address=stop.address,
                duration_minutes=stop.duration_minutes,
                arrival_time=stop.arrival_time,
                end_time=stop.end_time,
                transportation=stop.transportation,
                notes=stop.notes,
                lat=stop.lat,
                lng=stop.lng,
                google_place_id=stop.google_place_id,
            )
        )
    db.add(new_day)
    await _touch_trip(db, trip.id)
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after add_day")
    return _serialize(refreshed)


@router.delete("/{trip_id}/days/{day_id}", response_model=TripOut)
async def delete_day(
    trip_id: UUID,
    day_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    day = await _load_day(db, trip_id, day_id, user.id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    await db.delete(day)
    await _touch_trip(db, trip_id)
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after delete_day")
    return _serialize(refreshed)


@router.post(
    "/{trip_id}/days/{day_id}/stops",
    response_model=TripOut,
    status_code=status.HTTP_201_CREATED,
)
async def add_stop(
    trip_id: UUID,
    day_id: UUID,
    payload: TripStopCreateBody,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    day = await _load_day(db, trip_id, day_id, user.id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    body = payload.model_dump()
    requested = body.pop("sort_order", None)
    if requested is None:
        next_sort = max((s.sort_order for s in day.stops), default=-1) + 1
        new_stop = TripStop(trip_day_id=day.id, sort_order=next_sort, **body)
    else:
        # Shift existing stops at >= requested up by 1 in a SINGLE UPDATE so
        # UNIQUE(trip_day_id, sort_order) is checked once at end-of-statement.
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
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after add_stop")
    return _serialize(refreshed)


@router.patch("/{trip_id}/stops/{stop_id}", response_model=TripOut)
async def update_stop(
    trip_id: UUID,
    stop_id: UUID,
    payload: TripStopUpdate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    stop = await _load_stop(db, trip_id, stop_id, user.id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    for key, value in payload.model_dump(exclude_unset=True).items():
        setattr(stop, key, value)
    await _touch_trip(db, trip_id)
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update_stop")
    return _serialize(refreshed)


@router.delete("/{trip_id}/stops/{stop_id}", response_model=TripOut)
async def delete_stop(
    trip_id: UUID,
    stop_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    stop = await _load_stop(db, trip_id, stop_id, user.id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    # Per design: do NOT compact remaining sort_orders on delete. Gaps are
    # fine; the relationship's `order_by="TripStop.sort_order"` handles it.
    await db.delete(stop)
    await _touch_trip(db, trip_id)
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after delete_stop")
    return _serialize(refreshed)


@router.patch(
    "/{trip_id}/days/{day_id}/reorder", response_model=TripOut
)
async def reorder_stops(
    trip_id: UUID,
    day_id: UUID,
    payload: StopReorderBody,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    day = await _load_day(db, trip_id, day_id, user.id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    current_ids = {s.id for s in day.stops}
    body_ids = {item.stop_id for item in payload.order}
    if current_ids != body_ids:
        raise HTTPException(status_code=409, detail="stop set out of sync")
    body_orders = sorted(item.sort_order for item in payload.order)
    if body_orders != list(range(len(body_orders))):
        raise HTTPException(
            status_code=400,
            detail="sort_order values must be 0..N-1 contiguous",
        )
    new_order = {item.stop_id: item.sort_order for item in payload.order}
    # Single CASE-WHEN UPDATE — UNIQUE(trip_day_id, sort_order) is checked
    # at end-of-statement, so the position swap has no intermediate violation.
    await db.execute(
        update(TripStop)
        .where(TripStop.trip_day_id == day.id)
        .values(sort_order=case(new_order, value=TripStop.id))
    )
    await _touch_trip(db, trip_id)
    await db.commit()
    refreshed = await _load_trip(db, trip_id, user.id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after reorder")
    return _serialize(refreshed)


@router.put("/{trip_id}", response_model=TripOut)
async def replace_trip(
    trip_id: UUID,
    payload: TripCreate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    """Full replacement of a trip's days/stops. Same body shape as POST /trips,
    so the Flutter `ItineraryDraft.toCreatePayload` is reusable. The trip's id,
    created_at, and any associated conversation are preserved; everything else
    is wiped and rebuilt. Stop ids change on every replace — the response is
    the new source of truth."""
    updated = await replace_trip_from_payload(
        db,
        trip_id=trip_id,
        user_id=user.id,
        payload=payload.model_dump(mode="json"),
    )
    await db.commit()
    return _serialize(updated)


@router.post("/{trip_id}/conversation")
async def get_or_create_conversation_for_trip(
    trip_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    """Idempotent: return the (existing or newly-created) chat conversation
    bound to this trip. The Flutter `Refine with Buddy` button calls this
    eagerly so the chat screen can hydrate prior history before sending the
    first refine message."""
    trip = await _load_trip(db, trip_id, user.id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    conv = await get_or_create_refine_conversation(
        db, trip=trip, user_id=user.id
    )
    await db.commit()
    return {
        "id": str(conv.id),
        "title": conv.title,
        "trip_id": str(conv.trip_id) if conv.trip_id else None,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
    }
