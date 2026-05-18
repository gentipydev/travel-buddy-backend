"""Routed polylines via Google Directions, with a coords+mode keyed cache.

Cache rounds lat/lng to 4 decimals (~10m precision) so trips that share
endpoints — "near Berat castle" → "Berat town center" — reuse the same
polyline regardless of micro-differences in geocoding.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from typing import Literal

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import DirectionsCache
from app.domain.trips.geo import haversine_km
from app.integrations._retry import http_get_with_retry


logger = logging.getLogger(__name__)


_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
_ROADS_URL = "https://roads.googleapis.com/v1/nearestRoads"
_HTTP_TIMEOUT = 5.0
_CACHE_TTL = timedelta(days=90)
# When the first Directions call returns no path, both stop coords are
# snapped to their nearest road and Directions is retried. We reject the
# retry if either endpoint moved farther than this — the stop genuinely
# isn't near a routable road, and a polyline that visibly starts >250 m
# away from its marker is worse than no polyline.
_MAX_SNAP_METERS = 250.0


# Module-level singleton — same pattern as `google_places.py`. Keeps
# the connection pool warm across the per-stop directions fan-out and
# matches the Anthropic / Cohere client lifecycle.
_client_singleton = httpx.AsyncClient(timeout=_HTTP_TIMEOUT)


def _http_client() -> httpx.AsyncClient:
    """Indirection so tests can swap in `httpx.MockTransport`.

    Production returns the singleton; the `mock_http` conftest fixture
    monkeypatches this to return fresh clients with `MockTransport`.
    Callers must NOT use `async with _http_client() as client:` — that
    would close the singleton on exit.
    """
    return _client_singleton


def _cache_key(
    from_lat: float,
    from_lng: float,
    to_lat: float,
    to_lng: float,
    mode: str,
) -> str:
    return (
        f"{round(from_lat, 4)},{round(from_lng, 4)}"
        f"->{round(to_lat, 4)},{round(to_lng, 4)}:{mode}"
    )


async def get_route(
    db: AsyncSession,
    from_lat: float,
    from_lng: float,
    to_lat: float,
    to_lng: float,
    mode: Literal["driving", "walking"] = "driving",
) -> DirectionsCache | None:
    """Return a cached or freshly-fetched route. None on disabled / failure.

    On a first-attempt failure (ZERO_RESULTS, timeout, etc.) the endpoints
    are snapped to their nearest road via the Roads API and Directions is
    retried. The cache key stays anchored to the *original* coords so a
    successful retry caches under the caller's coords (next call short-
    circuits to cache without re-snapping).
    """
    started = time.perf_counter()
    key = _cache_key(from_lat, from_lng, to_lat, to_lng, mode)

    if not settings.google_api_key or not settings.google_directions_enabled:
        _log_done(key, "missing", started)
        return None

    cached = await _read_cache(db, key)
    if cached is not None:
        _log_done(key, "cache", started)
        return cached

    fresh = await _fetch_and_cache(
        db,
        key=key,
        from_lat=from_lat,
        from_lng=from_lng,
        to_lat=to_lat,
        to_lng=to_lng,
        mode=mode,
    )
    if fresh is not None:
        _log_done(key, "fresh", started)
        return fresh

    snapped = await _snap_to_roads([(from_lat, from_lng), (to_lat, to_lng)])
    if snapped is None:
        _log_done(key, "missing_snap_failed", started)
        return None
    (snap_from_lat, snap_from_lng), (snap_to_lat, snap_to_lng) = snapped
    from_drift_m = (
        haversine_km(from_lat, from_lng, snap_from_lat, snap_from_lng) * 1000.0
    )
    to_drift_m = (
        haversine_km(to_lat, to_lng, snap_to_lat, snap_to_lng) * 1000.0
    )
    if from_drift_m > _MAX_SNAP_METERS or to_drift_m > _MAX_SNAP_METERS:
        _log_done(key, "missing_snap_too_far", started)
        return None

    retry = await _fetch_and_cache(
        db,
        key=key,
        from_lat=snap_from_lat,
        from_lng=snap_from_lng,
        to_lat=snap_to_lat,
        to_lng=snap_to_lng,
        mode=mode,
    )
    _log_done(
        key,
        "fresh_after_snap" if retry else "missing_after_snap",
        started,
    )
    return retry


async def _snap_to_roads(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]] | None:
    """Snap each (lat, lng) to its nearest road via Google Roads API.

    Returns coords in the same order as [points], or None on any failure
    (Roads API disabled in GCP, network error, partial response).
    Uses the same API key as Directions; a 403 here typically means the
    Roads API isn't enabled for this key.
    """
    if not points or not settings.google_api_key:
        return None
    points_param = "|".join(f"{lat},{lng}" for lat, lng in points)
    client = _http_client()
    try:
        resp = await http_get_with_retry(
            client,
            _ROADS_URL,
            params={
                "points": points_param,
                "key": settings.google_api_key,
            },
        )
        if resp.status_code != 200:
            logger.warning(
                "google_api=roads_nearest status=%s", resp.status_code
            )
            return None
        data = resp.json()
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            "google_api=roads_nearest error=%s", type(exc).__name__
        )
        return None
    except Exception:
        logger.exception("google_api=roads_nearest unexpected")
        return None

    snapped = data.get("snappedPoints") or []
    # Build originalIndex → (lat, lng). Fail closed if any input is missing —
    # Directions needs both endpoints.
    by_idx: dict[int, tuple[float, float]] = {}
    for entry in snapped:
        idx = entry.get("originalIndex")
        loc = entry.get("location") or {}
        lat = loc.get("latitude")
        lng = loc.get("longitude")
        if idx is None or lat is None or lng is None:
            continue
        by_idx[idx] = (lat, lng)
    if len(by_idx) < len(points):
        return None
    return [by_idx[i] for i in range(len(points))]


async def _read_cache(
    db: AsyncSession, key: str
) -> DirectionsCache | None:
    now = datetime.now(timezone.utc)
    result = await db.execute(
        select(DirectionsCache).where(
            DirectionsCache.cache_key == key,
            DirectionsCache.expires_at > now,
        )
    )
    return result.scalar_one_or_none()


async def _fetch_and_cache(
    db: AsyncSession,
    *,
    key: str,
    from_lat: float,
    from_lng: float,
    to_lat: float,
    to_lng: float,
    mode: str,
) -> DirectionsCache | None:
    client = _http_client()
    try:
        resp = await http_get_with_retry(
            client,
            _DIRECTIONS_URL,
            params={
                "origin": f"{from_lat},{from_lng}",
                "destination": f"{to_lat},{to_lng}",
                "mode": mode,
                "key": settings.google_api_key,
            },
        )
        if resp.status_code == 429:
            # Reached only when retries are exhausted (policy tries 429
            # up to 3 times). Treat as quota_exceeded for visibility.
            logger.warning(
                "google_api=directions_get cache_key=%s status=quota_exceeded",
                key,
            )
            return None
        if resp.status_code != 200:
            logger.warning(
                "google_api=directions_get cache_key=%s status=%s",
                key,
                resp.status_code,
            )
            return None
        data = resp.json()
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            "google_api=directions_get cache_key=%s error=%s",
            key,
            type(exc).__name__,
        )
        return None
    except Exception:
        logger.exception(
            "google_api=directions_get cache_key=%s unexpected", key
        )
        return None

    routes = data.get("routes") or []
    if not routes:
        # ZERO_RESULTS — Directions has no path (e.g. walking across a sea).
        # Don't cache; the orchestrator will draw a straight line client-side.
        logger.info(
            "google_api=directions_get cache_key=%s api_status=%s",
            key,
            data.get("status"),
        )
        return None

    overview = (routes[0].get("overview_polyline") or {}).get("points")
    if not overview:
        return None
    legs = routes[0].get("legs") or []
    distance_m = (legs[0].get("distance") or {}).get("value") if legs else None
    duration_s = (legs[0].get("duration") or {}).get("value") if legs else None

    return await _upsert_cache(
        db,
        key=key,
        encoded_polyline=overview,
        distance_meters=distance_m,
        duration_seconds=duration_s,
        travel_mode=mode,
    )


async def _upsert_cache(
    db: AsyncSession,
    *,
    key: str,
    encoded_polyline: str,
    distance_meters: int | None,
    duration_seconds: int | None,
    travel_mode: str,
) -> DirectionsCache:
    now = datetime.now(timezone.utc)
    expires = now + _CACHE_TTL
    stmt = (
        pg_insert(DirectionsCache)
        .values(
            cache_key=key,
            encoded_polyline=encoded_polyline,
            distance_meters=distance_meters,
            duration_seconds=duration_seconds,
            travel_mode=travel_mode,
            fetched_at=now,
            expires_at=expires,
        )
        .on_conflict_do_update(
            index_elements=["cache_key"],
            set_={
                "encoded_polyline": encoded_polyline,
                "distance_meters": distance_meters,
                "duration_seconds": duration_seconds,
                "travel_mode": travel_mode,
                "fetched_at": now,
                "expires_at": expires,
            },
        )
    )
    await db.execute(stmt)
    # Re-read so the caller gets a managed instance with the upserted state.
    result = await db.execute(
        select(DirectionsCache).where(DirectionsCache.cache_key == key)
    )
    return result.scalar_one()


def _log_done(key: str, source: str, started: float) -> None:
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "google_api=directions_get cache_key=%s source=%s latency_ms=%s",
        key,
        source,
        elapsed_ms,
    )
