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
import asyncio
import logging
from collections.abc import Iterable

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.engine import get_session
from app.db.models import Artifact
from app.integrations.supabase_storage import (
    ARTIFACT_PHOTOS_BUCKET,
    PLACE_PHOTOS_BUCKET,
    THUMBS_FOLDER,
    list_image_names,
    public_url,
    thumb_path,
)

router = APIRouter(tags=["images"])
logger = logging.getLogger(__name__)


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
    # Small copies for the gallery grid, same order as `photos`. An entry is
    # the full-size URL when that photo has no small copy yet.
    thumbs: list[str]


class ArtifactOut(BaseModel):
    """Mirrors the mobile `ArtifactData` shape so the JSON can be deserialized
    directly into the existing model. `image_small` / `image_medium` point at
    the small copy in `thumbs/` when there is one, else at the full image."""
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


async def _existing_thumbs(bucket: str, folders: Iterable[str]) -> set[str]:
    """Paths of the small copies that exist under `{folder}/thumbs/`, one
    listing per folder. A failed listing just means full-size URLs are
    served, so it is logged and skipped rather than failing the request."""
    folders = sorted(set(folders))
    results = await asyncio.gather(
        *(list_image_names(bucket, f"{folder}/{THUMBS_FOLDER}") for folder in folders),
        return_exceptions=True,
    )
    existing: set[str] = set()
    for folder, names in zip(folders, results, strict=True):
        if isinstance(names, BaseException):
            logger.warning("thumb listing failed for %s/%s: %r", bucket, folder, names)
            continue
        existing.update(f"{folder}/{THUMBS_FOLDER}/{name}" for name in names)
    return existing


async def _artifact_thumbs(rows: Iterable[Artifact]) -> set[str]:
    return await _existing_thumbs(
        ARTIFACT_PHOTOS_BUCKET, (row.image_path.rpartition("/")[0] for row in rows)
    )


async def _to_artifact_out(row: Artifact, thumbs: set[str]) -> ArtifactOut:
    url = await public_url(ARTIFACT_PHOTOS_BUCKET, row.image_path)
    small_path = thumb_path(row.image_path)
    small = await public_url(ARTIFACT_PHOTOS_BUCKET, small_path) if small_path in thumbs else url
    return ArtifactOut(
        id=row.id,
        title=row.title,
        image=url,
        image_small=small,
        image_medium=small,
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
    sorted by filename, plus each photo's small copy from `thumbs/`. Empty
    arrays if the folder hasn't been seeded yet — not an error, the app
    falls back to its bundled photos."""
    _validate_slug(slug)
    folder = f"destinations/{slug}"
    names, thumbs = await asyncio.gather(
        list_image_names(PLACE_PHOTOS_BUCKET, folder),
        _existing_thumbs(PLACE_PHOTOS_BUCKET, [folder]),
    )
    photos: list[str] = []
    small: list[str] = []
    for name in names:
        url = await public_url(PLACE_PHOTOS_BUCKET, f"{folder}/{name}")
        small_path = thumb_path(f"{folder}/{name}")
        photos.append(url)
        small.append(
            await public_url(PLACE_PHOTOS_BUCKET, small_path) if small_path in thumbs else url
        )
    return DestinationPhotos(slug=slug, photos=photos, thumbs=small)


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

    thumbs = await _artifact_thumbs(rows)
    highlights: list[ArtifactOut] = []
    collectibles: list[ArtifactOut] = []
    for row in rows:
        out = await _to_artifact_out(row, thumbs)
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
    return await _to_artifact_out(row, await _artifact_thumbs([row]))


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
    thumbs = await _artifact_thumbs(rows)
    return [await _to_artifact_out(r, thumbs) for r in rows]
