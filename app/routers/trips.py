import logging
from datetime import datetime
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select, tuple_
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.db.engine import get_session
from app.db.models import Trip, TripDay, TripStop, User
from app.deps import require_local_user
from app.domain.trips import (
    apply_add_day,
    apply_delete_day,
    apply_delete_stop,
    apply_insert_stop,
    apply_reorder_stops,
    apply_update_stop,
    apply_update_trip,
    get_or_create_refine_conversation,
    replace_trip_from_payload,
    serialize_trip,
)
from app.domain.trips.repository import _load_trip, _serialize
from app.schemas import (
    PaginatedResponse,
    StopReorderBody,
    TripCreate,
    TripDayCreateBody,
    TripOut,
    TripStopCreateBody,
    TripStopUpdate,
    TripUpdate,
)

logger = logging.getLogger(__name__)


def _parse_cursor(cursor: str) -> tuple[datetime, UUID]:
    """Split a `"{updated_at_iso}|{uuid}"` cursor into its two parts.

    Keyset pagination needs both halves: `updated_at` alone collides under
    same-second updates; `id` alone breaks the visible ordering. The pair
    is opaque to the client — we don't promise the format won't change."""
    try:
        ts_str, id_str = cursor.split("|", 1)
        return datetime.fromisoformat(ts_str), UUID(id_str)
    except (ValueError, AttributeError) as exc:
        raise HTTPException(
            status_code=400, detail=f"invalid cursor: {exc}"
        ) from exc

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
        return await serialize_trip(db, refreshed)
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
        ) from exc


@router.get(
    "",
    response_model=PaginatedResponse[TripOut],
    responses={
        400: {"description": "Invalid cursor format"},
        422: {"description": "limit out of range (must be 1..100)"},
    },
)
async def list_trips(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = Query(
        None,
        description=(
            "Opaque cursor from a previous response's `next_cursor`. "
            "Omit for the first page."
        ),
    ),
):
    """Keyset-paginated list of the caller's trips, newest first.

    Ordering is `(updated_at DESC, id DESC)` so a same-second-updated tie
    still has a deterministic page boundary. Fetches `limit + 1` rows and
    derives `next_cursor` from the (limit+1)th — when fewer than that come
    back, `next_cursor` is None and the caller has the last page.
    """
    stmt = (
        select(Trip)
        .where(Trip.user_id == user.id)
        .options(
            selectinload(Trip.days).selectinload(TripDay.stops),
            selectinload(Trip.routes),
        )
        .order_by(Trip.updated_at.desc(), Trip.id.desc())
    )
    if cursor:
        ts, last_id = _parse_cursor(cursor)
        # Row-constructor comparison: PostgreSQL evaluates this lexicographically,
        # so a row "ties" on updated_at only if its id also breaks the comparison
        # the same way — no rows skipped at second-boundaries, no rows repeated.
        stmt = stmt.where(
            tuple_(Trip.updated_at, Trip.id) < tuple_(ts, last_id)
        )
    rows = (await db.execute(stmt.limit(limit + 1))).unique().scalars().all()
    next_cursor: str | None = None
    if len(rows) > limit:
        rows = rows[:limit]
        last = rows[-1]
        next_cursor = f"{last.updated_at.isoformat()}|{last.id}"
    return PaginatedResponse[TripOut](
        items=[_serialize(t) for t in rows], next_cursor=next_cursor,
    )


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
    return await serialize_trip(db, updated)


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
