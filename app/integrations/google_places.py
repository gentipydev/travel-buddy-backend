"""Resolve a stop's `google_place_id` to display data (photo, rating, address).

Three tiers, in order:
    1. Curated `places` table (vetted Albanian destinations) — local-only,
       always wins. Kill switches do not gate this tier.
    2. `places_cache` table — Google Place Details responses kept for 90 days.
    3. Google Place Details API — the miss path; result is written to cache.

Saves never fail because of this module. Every error path logs and returns
None so the caller can proceed without enrichment. The `source` field on
`EnrichmentResult` is the metric for both curated coverage and cache-hit
rate.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import Place, PlaceCache
from app.integrations._retry import http_get_with_retry


def _is_in_albania(lat: float, lng: float) -> bool:
    # Lazy import to break the circular dep:
    # `app.domain.trips.__init__` eagerly imports `enrichment`, which imports
    # back from this module. A module-level `from app.domain.trips.geo import …`
    # evaluates the package init and trips the cycle (visible once the test
    # files actually run, post the conftest anyio-mark fix).
    from app.domain.trips.geo import is_in_albania

    return is_in_albania(lat, lng)


logger = logging.getLogger(__name__)


_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
_PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"
_TEXT_SEARCH_URL = "https://maps.googleapis.com/maps/api/place/textsearch/json"
_DETAILS_FIELDS = (
    "place_id,name,formatted_address,geometry,rating,user_ratings_total,photos,types"
)
_HTTP_TIMEOUT = 5.0
_MAX_PHOTOS = 5   # primary + 4 additional; resolving more doubles spend
_CACHE_TTL = timedelta(days=90)


@dataclass
class EnrichmentResult:
    photo_url: str | None
    # Photos beyond the primary, in the order Google returned them. Empty for
    # curated rows with at most one photo or when the API returned a single
    # reference. The caller composes `[photo_url] + additional_photos` (capped)
    # to write `TripStop.photos`.
    additional_photos: list[str]
    rating: float | None
    formatted_address: str | None
    lat: float | None
    lng: float | None
    source: Literal["curated", "cache", "fresh"]


# Module-level singleton mirrors `app/llm/anthropic_client.py` and
# `app/integrations/cohere_client.py`: keep one connection pool warm
# instead of paying ~30 ms of TLS + client init per enrichment call.
# Lives for the lifetime of the process; FastAPI's shutdown doesn't
# explicitly close it — httpx finalizes on GC, which is fine for a
# read-only outbound client (no in-flight writes to flush).
_client_singleton = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)


def _http_client() -> httpx.AsyncClient:
    """Indirection so tests can swap in `httpx.MockTransport`.

    Production returns the singleton; the `mock_http` conftest fixture
    monkeypatches this to return fresh clients with `MockTransport`.
    Callers must NOT use `async with _http_client() as client:` — that
    would close the singleton on exit.
    """
    return _client_singleton


async def enrich_stop(
    db: AsyncSession,
    google_place_id: str,
) -> EnrichmentResult | None:
    """Curated → cache → API. Returns None only when all three miss."""
    started = time.perf_counter()

    # Tier 1: curated.
    curated = await _read_curated(db, google_place_id)
    if curated is not None:
        _log_done(google_place_id, "curated", started)
        return curated

    # Kill switch only gates tiers 2-3. Curated is local-only.
    if not settings.google_api_key or not settings.google_places_enabled:
        _log_done(google_place_id, "missing", started)
        return None

    # Tier 2: cache.
    cached = await _read_cache(db, google_place_id)
    if cached is not None:
        _log_done(google_place_id, "cache", started)
        return cached

    # Tier 3: Google Place Details + photo redirect resolution.
    fresh = await _fetch_and_cache(db, google_place_id)
    _log_done(google_place_id, "fresh" if fresh else "missing", started)
    return fresh


async def _read_curated(
    db: AsyncSession, google_place_id: str
) -> EnrichmentResult | None:
    result = await db.execute(
        select(Place).where(Place.google_place_id == google_place_id)
    )
    place = result.scalar_one_or_none()
    if place is None:
        return None
    photos = list(place.photos or [])
    address = (
        f"{place.name}, {place.city}" if place.city else place.name
    )
    return EnrichmentResult(
        photo_url=photos[0] if photos else None,
        additional_photos=photos[1:],
        rating=place.rating,
        formatted_address=address,
        lat=place.latitude,
        lng=place.longitude,
        source="curated",
    )


async def _read_cache(
    db: AsyncSession, google_place_id: str
) -> EnrichmentResult | None:
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(PlaceCache).where(
            PlaceCache.google_place_id == google_place_id,
            PlaceCache.expires_at > now,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return EnrichmentResult(
        photo_url=row.primary_photo_url,
        additional_photos=list(row.additional_photo_urls or []),
        rating=row.rating,
        formatted_address=row.formatted_address,
        lat=row.lat,
        lng=row.lng,
        source="cache",
    )


async def _fetch_and_cache(
    db: AsyncSession, google_place_id: str
) -> EnrichmentResult | None:
    client = _http_client()
    try:
        details_resp = await http_get_with_retry(
            client,
            _DETAILS_URL,
            params={
                "place_id": google_place_id,
                "fields": _DETAILS_FIELDS,
                "key": settings.google_api_key,
            },
        )
        if details_resp.status_code == 429:
            # Reached only when retries are exhausted — the retry policy
            # tries 429 up to 3 times. Treat as quota_exceeded for
            # operator visibility.
            logger.warning(
                "google_api=places_enrich place_id=%s status=quota_exceeded",
                google_place_id,
            )
            return None
        if details_resp.status_code != 200:
            logger.warning(
                "google_api=places_enrich place_id=%s status=%s",
                google_place_id,
                details_resp.status_code,
            )
            return None
        data = details_resp.json()
        if data.get("status") not in ("OK", None):
            logger.warning(
                "google_api=places_enrich place_id=%s api_status=%s",
                google_place_id,
                data.get("status"),
            )
            return None
        result = data.get("result") or {}
        if not result:
            return None
        photo_refs = [
            p.get("photo_reference")
            for p in (result.get("photos") or [])[:_MAX_PHOTOS]
            if p.get("photo_reference")
        ]
        primary, additional = await _resolve_photos(client, photo_refs)
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            "google_api=places_enrich place_id=%s error=%s",
            google_place_id,
            type(exc).__name__,
        )
        return None
    except Exception:
        logger.exception(
            "google_api=places_enrich place_id=%s unexpected", google_place_id
        )
        return None

    geom = (result.get("geometry") or {}).get("location") or {}
    lat = geom.get("lat")
    lng = geom.get("lng")
    if lat is None or lng is None:
        # No coords means we can't put a marker on the map. Don't cache an
        # unusable row — return None and let the next call retry.
        return None

    if not _is_in_albania(lat, lng):
        # Refuse to seed the cache with a non-AL row — once cached, a future
        # caller hitting the same Claude-generated place_id would still pull
        # the wrong country. Returning None drops the stop's enrichment.
        logger.warning(
            "google_api=places_enrich place_id=%s dropped=out_of_bounds "
            "lat=%s lng=%s",
            google_place_id, lat, lng,
        )
        return None

    formatted = result.get("formatted_address") or result.get("name") or ""
    name = result.get("name") or formatted

    await _upsert_cache(
        db,
        google_place_id=google_place_id,
        name=name,
        formatted_address=formatted,
        lat=lat,
        lng=lng,
        rating=result.get("rating"),
        user_ratings_total=result.get("user_ratings_total"),
        primary_photo_url=primary,
        additional_photo_urls=additional,
        types=result.get("types") or [],
        raw_response=result,
    )

    return EnrichmentResult(
        photo_url=primary,
        additional_photos=additional,
        rating=result.get("rating"),
        formatted_address=formatted,
        lat=lat,
        lng=lng,
        source="fresh",
    )


async def _resolve_photos(
    client: httpx.AsyncClient, photo_refs: list[str]
) -> tuple[str | None, list[str]]:
    """Convert photo_reference strings to stable redirect-target URLs.

    Each `place/photo` endpoint returns a 302 to a googleusercontent.com URL
    that's stable for months — much longer than the photo_reference itself.
    Failed resolutions are logged and skipped (the place is still useful).
    """
    resolved: list[str] = []
    for ref in photo_refs:
        try:
            # 302 is the expected success — _is_retryable_status only
            # retries on 429/5xx, so the redirect path is unaffected.
            resp = await http_get_with_retry(
                client,
                _PHOTO_URL,
                params={
                    "photo_reference": ref,
                    "maxwidth": 1000,
                    "key": settings.google_api_key,
                },
                follow_redirects=False,
            )
        except (httpx.RequestError, httpx.TimeoutException) as exc:
            logger.warning(
                "google_api=places_photo error=%s", type(exc).__name__
            )
            continue
        if resp.status_code in (301, 302, 303, 307, 308):
            target = resp.headers.get("location")
            if target:
                resolved.append(target)
            continue
        logger.warning(
            "google_api=places_photo status=%s", resp.status_code
        )

    if not resolved:
        return None, []
    return resolved[0], resolved[1:]


async def _upsert_cache(
    db: AsyncSession,
    *,
    google_place_id: str,
    name: str,
    formatted_address: str | None,
    lat: float,
    lng: float,
    rating: float | None,
    user_ratings_total: int | None,
    primary_photo_url: str | None,
    additional_photo_urls: list[str],
    types: list[str],
    raw_response: dict,
) -> None:
    now = datetime.now(timezone.utc)
    expires = now + _CACHE_TTL
    stmt = (
        pg_insert(PlaceCache)
        .values(
            google_place_id=google_place_id,
            name=name,
            formatted_address=formatted_address,
            lat=lat,
            lng=lng,
            rating=rating,
            user_ratings_total=user_ratings_total,
            primary_photo_url=primary_photo_url,
            additional_photo_urls=additional_photo_urls,
            types=types,
            raw_response=raw_response,
            fetched_at=now,
            expires_at=expires,
        )
        .on_conflict_do_update(
            index_elements=["google_place_id"],
            set_={
                "name": name,
                "formatted_address": formatted_address,
                "lat": lat,
                "lng": lng,
                "rating": rating,
                "user_ratings_total": user_ratings_total,
                "primary_photo_url": primary_photo_url,
                "additional_photo_urls": additional_photo_urls,
                "types": types,
                "raw_response": raw_response,
                "fetched_at": now,
                "expires_at": expires,
            },
        )
    )
    await db.execute(stmt)


def _log_done(google_place_id: str, source: str, started: float) -> None:
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "google_api=places_enrich place_id=%s source=%s latency_ms=%s",
        google_place_id,
        source,
        elapsed_ms,
    )


async def find_place_id_by_name(
    *,
    name: str,
    city: str | None = None,
) -> tuple[str, float, float] | None:
    """Resolve a free-text place query to (place_id, lat, lng).

    Used by trip enrichment when Claude produces a stop name without a
    `google_place_id`. The caller persists the place_id on the stop so
    subsequent runs hit the curated/cache path and avoid another text
    search. Returns None when the kill switch is off, the request fails,
    or there are no results.
    """
    if not settings.google_api_key or not settings.google_places_enabled:
        return None

    # Always append ", Albania" so the model's city alone can't drift to a
    # same-named locality elsewhere. `region=al` is a soft ranking bias on
    # the Text Search; the bbox post-filter below is the hard guard.
    query = f"{name}, {city}, Albania" if city else f"{name}, Albania"
    started = time.perf_counter()
    client = _http_client()
    try:
        resp = await http_get_with_retry(
            client,
            _TEXT_SEARCH_URL,
            params={
                "query": query,
                "region": "al",
                "key": settings.google_api_key,
            },
        )
        if resp.status_code == 429:
            logger.warning(
                "google_api=places_textsearch query=%r status=quota_exceeded",
                query,
            )
            return None
        if resp.status_code != 200:
            logger.warning(
                "google_api=places_textsearch query=%r status=%s",
                query,
                resp.status_code,
            )
            return None
        data = resp.json()
        api_status = data.get("status")
        if api_status == "ZERO_RESULTS":
            _log_textsearch_done(query, "no_results", started)
            return None
        if api_status not in ("OK", None):
            logger.warning(
                "google_api=places_textsearch query=%r api_status=%s",
                query,
                api_status,
            )
            return None
        results = data.get("results") or []
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            "google_api=places_textsearch query=%r error=%s",
            query,
            type(exc).__name__,
        )
        return None
    except Exception:
        logger.exception(
            "google_api=places_textsearch query=%r unexpected", query
        )
        return None

    if not results:
        _log_textsearch_done(query, "no_results", started)
        return None
    top = results[0]
    place_id = top.get("place_id")
    geom = (top.get("geometry") or {}).get("location") or {}
    lat = geom.get("lat")
    lng = geom.get("lng")
    if not place_id or lat is None or lng is None:
        _log_textsearch_done(query, "no_results", started)
        return None

    if not _is_in_albania(lat, lng):
        # Hard guard against the unbounded-text-search failure: refuse to
        # return coords / place_id outside the Albania bbox. The caller
        # leaves the stop un-geo'd, which is correct fail-safe behavior.
        logger.warning(
            "google_api=places_textsearch query=%r dropped=out_of_bounds "
            "lat=%s lng=%s",
            query, lat, lng,
        )
        _log_textsearch_done(query, "out_of_bounds", started)
        return None

    _log_textsearch_done(query, "fresh", started)
    return place_id, float(lat), float(lng)


def _log_textsearch_done(query: str, source: str, started: float) -> None:
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "google_api=places_textsearch query=%r source=%s latency_ms=%s",
        query,
        source,
        elapsed_ms,
    )
