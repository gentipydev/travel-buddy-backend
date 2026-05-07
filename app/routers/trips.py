import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

logger = logging.getLogger(__name__)

from app.db.engine import get_session
from app.db.models import Trip, TripDay, TripStop, User
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
    apply_add_day,
    apply_delete_day,
    apply_delete_stop,
    apply_insert_stop,
    apply_reorder_stops,
    apply_update_stop,
    apply_update_trip,
    get_or_create_refine_conversation,
    replace_trip_from_payload,
)

# TODO(tests): pytest scaffolding deferred to a follow-up PR — see
# manual curl smoke test recipes in the trip editing plan.

router = APIRouter(prefix="/trips", tags=["trips"])


@router.post("", response_model=TripOut, status_code=status.HTTP_201_CREATED)
async def create_trip(
    payload: TripCreate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    # Snapshot scalar fields before any DB op so the except branch can log
    # without triggering a lazy attribute load on an expired/failed session
    # (that would raise `MissingGreenlet` and mask the real exception).
    user_id_for_log = user.id
    # Wrap the WHOLE handler so any exception (commit, _load_trip,
    # _serialize, ORM relationship access, lazy-load mistake) surfaces a
    # JSON response with the exception type+message instead of an opaque
    # plain-text "Internal Server Error". Full traceback still lands in
    # server logs via logger.exception.
    try:
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
        logger.info(
            "trip.created",
            extra={
                "trip_id": str(trip.id),
                "local_user_id": user.id,
                "days": len(payload.days),
                "stops": sum(len(d.stops) for d in payload.days),
            },
        )

        refreshed = await _load_trip(db, trip.id, user.id)
        if refreshed is None:
            # Should be impossible — we just committed it. Defensive.
            raise HTTPException(
                status_code=500, detail="Trip not found after create"
            )
        return _serialize(refreshed)
    except HTTPException:
        # Already a structured FastAPI error — let it through unchanged.
        raise
    except Exception as exc:
        try:
            await db.rollback()
        except Exception:
            pass
        logger.exception(
            "create_trip failed user_id=%s days=%d",
            user_id_for_log,
            len(payload.days),
        )
        raise HTTPException(
            status_code=500,
            detail=f"create_trip failed: {type(exc).__name__}: {exc}",
        )


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
    logger.info(
        "trip.deleted",
        extra={"trip_id": str(trip_id), "local_user_id": user.id},
    )
    return {"status": "deleted"}


@router.patch("/{trip_id}", response_model=TripOut)
async def update_trip(
    trip_id: UUID,
    payload: TripUpdate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    patch = payload.model_dump(exclude_unset=True)
    updated = await apply_update_trip(
        db,
        trip_id=trip_id,
        user_id=user.id,
        payload=patch,
    )
    await db.commit()
    logger.info(
        "trip.updated",
        extra={
            "trip_id": str(trip_id),
            "local_user_id": user.id,
            "fields": sorted(patch.keys()),
        },
    )
    return _serialize(updated)


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
    updated = await apply_add_day(
        db,
        trip_id=trip_id,
        user_id=user.id,
        payload=payload.model_dump(),
    )
    await db.commit()
    return _serialize(updated)


@router.delete("/{trip_id}/days/{day_id}", response_model=TripOut)
async def delete_day(
    trip_id: UUID,
    day_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    updated = await apply_delete_day(
        db, trip_id=trip_id, user_id=user.id, day_id=day_id
    )
    await db.commit()
    return _serialize(updated)


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
    updated = await apply_insert_stop(
        db,
        trip_id=trip_id,
        user_id=user.id,
        day_id=day_id,
        payload=payload.model_dump(),
    )
    await db.commit()
    return _serialize(updated)


@router.patch("/{trip_id}/stops/{stop_id}", response_model=TripOut)
async def update_stop(
    trip_id: UUID,
    stop_id: UUID,
    payload: TripStopUpdate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    updated = await apply_update_stop(
        db,
        trip_id=trip_id,
        user_id=user.id,
        stop_id=stop_id,
        payload=payload.model_dump(exclude_unset=True),
    )
    await db.commit()
    return _serialize(updated)


@router.delete("/{trip_id}/stops/{stop_id}", response_model=TripOut)
async def delete_stop(
    trip_id: UUID,
    stop_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    updated = await apply_delete_stop(
        db, trip_id=trip_id, user_id=user.id, stop_id=stop_id
    )
    await db.commit()
    return _serialize(updated)


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
    updated = await apply_reorder_stops(
        db,
        trip_id=trip_id,
        user_id=user.id,
        day_id=day_id,
        # exclude_unset so callers that omit the optional time fields don't
        # accidentally null them on the server (per-item arrival_time etc.
        # are only written when the caller explicitly included them).
        payload=payload.model_dump(mode="json", exclude_unset=True),
    )
    await db.commit()
    return _serialize(updated)


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
