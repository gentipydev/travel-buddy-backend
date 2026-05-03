from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import Trip, User
from app.deps import require_local_user
from app.schemas import TripCreate, TripOut


router = APIRouter(prefix="/trips", tags=["trips"])


def _serialize(trip: Trip) -> dict:
    return {
        "id": str(trip.id),
        "title": trip.title,
        "itinerary": dict(trip.itinerary or {}),
        "place_id": str(trip.place_id) if trip.place_id else None,
        "created_at": trip.created_at.isoformat() if trip.created_at else "",
        "updated_at": trip.updated_at.isoformat() if trip.updated_at else "",
    }


@router.post("", response_model=TripOut)
async def create_trip(
    payload: TripCreate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    place_uuid: UUID | None = None
    if payload.place_id:
        try:
            place_uuid = UUID(payload.place_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid place_id")

        existing = (await db.execute(
            select(Trip).where(
                Trip.user_id == user.id,
                Trip.place_id == place_uuid,
            )
        )).scalar_one_or_none()
        if existing is not None:
            return _serialize(existing)

    trip = Trip(
        user_id=user.id,
        place_id=place_uuid,
        title=payload.title,
        itinerary=payload.itinerary or {},
    )
    db.add(trip)
    await db.commit()
    await db.refresh(trip)
    return _serialize(trip)


@router.get("")
async def list_trips(
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(Trip)
        .where(Trip.user_id == user.id)
        .order_by(Trip.updated_at.desc())
    )
    trips = result.scalars().all()
    return [_serialize(t) for t in trips]


@router.delete("/by-place/{place_id}")
async def delete_trip_by_place(
    place_id: UUID,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    result = await db.execute(
        select(Trip).where(
            Trip.user_id == user.id,
            Trip.place_id == place_id,
        )
    )
    trip = result.scalar_one_or_none()
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    await db.delete(trip)
    await db.commit()
    return {"status": "deleted"}
