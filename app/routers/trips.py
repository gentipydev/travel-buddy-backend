from fastapi import APIRouter, Depends
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
        "created_at": trip.created_at.isoformat() if trip.created_at else "",
        "updated_at": trip.updated_at.isoformat() if trip.updated_at else "",
    }


@router.post("", response_model=TripOut)
async def create_trip(
    payload: TripCreate,
    user: User = Depends(require_local_user),
    db: AsyncSession = Depends(get_session),
):
    trip = Trip(
        user_id=user.id,
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
