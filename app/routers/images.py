"""Image manifest endpoints — the single source of truth for every remote
image rendered by the mobile app's destination flows.

Mobile no longer constructs Unsplash or MET URLs; it calls these endpoints
to discover destination photos and curated Albanian artifacts. Image bytes
are stored in Supabase (`place-photos/destinations/{slug}/` and the
`artifact-photos` bucket), so swapping content is a Storage upload, not a
mobile release.

Endpoints are public — `NetworkImage` cannot easily attach a Bearer token,
and these are non-sensitive cultural-heritage assets. The buckets
themselves are public-read.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import Depends

from app.db.engine import get_session
from app.db.models import Artifact
from app.integrations.supabase_storage import (
    PLACE_PHOTOS_BUCKET,
    get_artifact_image_url,
    list_bucket_folder,
)


router = APIRouter(tags=["images"])


# Slugs are validated against this set so that an unknown destination 404s
# at the API rather than triggering a wide-open Storage list call. Mirrors
# `lib/logic/data/destination_type.dart`.
DESTINATION_SLUGS: frozenset[str] = frozenset({
    "albanian_alps",
    "apollonia",
    "berat",
    "butrint",
    "gjirokaster",
    "korca",
    "kruja",
    "theth",
})


class DestinationPhotos(BaseModel):
    slug: str
    photos: list[str]


class ArtifactOut(BaseModel):
    """Mirrors the mobile `ArtifactData` shape so the JSON can be deserialized
    directly into the existing model. `image_small` / `image_medium` are
    reserved for future size variants — today they all return the same URL."""
    id: str
    title: str
    image: str
    image_small: str
    image_medium: str
    date: str
    period: str
    country: str
    culture: str
    medium: str
    dimension: str
    classification: str
    object_type: str
    object_begin_year: int
    object_end_year: int
    # Bundled-asset icon name for collectibles (e.g. 'jewelry'); empty for highlights.
    icon_name: str = ""


class ArtifactsByCategory(BaseModel):
    highlights: list[ArtifactOut]
    collectibles: list[ArtifactOut]


def _validate_slug(slug: str) -> str:
    if slug not in DESTINATION_SLUGS:
        raise HTTPException(status_code=404, detail=f"Unknown destination: {slug}")
    return slug


async def _to_artifact_out(row: Artifact) -> ArtifactOut:
    url = await get_artifact_image_url(row.image_path)
    return ArtifactOut(
        id=row.id,
        title=row.title,
        image=url,
        image_small=url,
        image_medium=url,
        date=row.date_text or "",
        period=row.period or "",
        country=row.country or "",
        culture=row.culture or "",
        medium=row.medium or "",
        dimension=row.dimension or "",
        classification=row.classification or "",
        object_type=row.object_type or "",
        object_begin_year=row.object_begin_year or 0,
        object_end_year=row.object_end_year or 0,
        icon_name=row.icon_name or "",
    )


@router.get("/destinations/{slug}/photos", response_model=DestinationPhotos)
async def list_destination_photos(slug: str) -> DestinationPhotos:
    """Photo gallery manifest. Lists `place-photos/destinations/{slug}/`,
    sorted by filename. Empty array if the folder hasn't been seeded yet —
    not an error, the gallery shows its existing spinner state."""
    _validate_slug(slug)
    photos = await list_bucket_folder(
        PLACE_PHOTOS_BUCKET, f"destinations/{slug}"
    )
    return DestinationPhotos(slug=slug, photos=photos)


@router.get(
    "/destinations/{slug}/artifacts",
    response_model=ArtifactsByCategory,
)
async def list_destination_artifacts(
    slug: str,
    db: AsyncSession = Depends(get_session),
) -> ArtifactsByCategory:
    _validate_slug(slug)
    rows = (await db.execute(
        select(Artifact)
        .where(Artifact.destination_slug == slug)
        .order_by(Artifact.category, Artifact.sort_order)
    )).scalars().all()

    highlights: list[ArtifactOut] = []
    collectibles: list[ArtifactOut] = []
    for row in rows:
        out = await _to_artifact_out(row)
        if row.category == "highlight":
            highlights.append(out)
        elif row.category == "collectible":
            collectibles.append(out)
    return ArtifactsByCategory(highlights=highlights, collectibles=collectibles)


@router.get("/artifacts/{artifact_id}", response_model=ArtifactOut)
async def get_artifact(
    artifact_id: str,
    db: AsyncSession = Depends(get_session),
) -> ArtifactOut:
    row = (await db.execute(
        select(Artifact).where(Artifact.id == artifact_id)
    )).scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"Artifact not found: {artifact_id}")
    return await _to_artifact_out(row)


@router.get(
    "/destinations/{slug}/artifacts/search",
    response_model=list[ArtifactOut],
)
async def search_destination_artifacts(
    slug: str,
    q: str = Query(min_length=1, max_length=200),
    limit: int = Query(default=50, ge=1, le=200),
    db: AsyncSession = Depends(get_session),
) -> list[ArtifactOut]:
    """Per-destination text search. `plainto_tsquery` accepts arbitrary
    user input safely (whitespace-separated terms ANDed). Ranking falls
    back to sort_order for ties so behavior is deterministic on small
    catalogs."""
    _validate_slug(slug)
    stmt = (
        select(Artifact)
        .where(
            Artifact.destination_slug == slug,
            text("search_text @@ plainto_tsquery('simple', :q)"),
        )
        .order_by(
            text("ts_rank(search_text, plainto_tsquery('simple', :q)) DESC"),
            Artifact.sort_order,
        )
        .limit(limit)
    )
    rows = (await db.execute(stmt, {"q": q})).scalars().all()
    return [await _to_artifact_out(r) for r in rows]
