from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import Place


async def detect_place(message: str, db: AsyncSession) -> Place | None:
    """Return the best Place whose name appears as a substring of `message`.

    Ranking: longer names first (more specific), then higher rating.
    Case-insensitive; non-ASCII names are compared in their lowercased form.
    """
    lowered = message.lower()
    rows = (await db.execute(select(Place))).scalars().all()
    best: Place | None = None
    for p in rows:
        if not p.name:
            continue
        if p.name.lower() not in lowered:
            continue
        if (
            best is None
            or len(p.name) > len(best.name)
            or (
                len(p.name) == len(best.name)
                and (p.rating or 0) > (best.rating or 0)
            )
        ):
            best = p
    return best


def to_card_dict(place: Place) -> dict:
    photos = list(place.photos or [])
    return {
        "id": str(place.id),
        "name": place.name,
        "city": place.city,
        "region": place.region,
        "tags": list(place.tags or []),
        "hero_image_url": photos[0] if photos else None,
        "rating": place.rating,
        "lat": place.latitude,
        "lng": place.longitude,
        "google_place_id": place.google_place_id,
    }
