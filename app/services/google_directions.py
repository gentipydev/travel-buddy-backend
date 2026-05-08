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


logger = logging.getLogger(__name__)


_DIRECTIONS_URL = "https://maps.googleapis.com/maps/api/directions/json"
_HTTP_TIMEOUT = 5.0
_CACHE_TTL = timedelta(days=90)


def _http_client() -> httpx.AsyncClient:
    """Indirection so tests can swap in `httpx.MockTransport`."""
    return httpx.AsyncClient(timeout=_HTTP_TIMEOUT)


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
    """Return a cached or freshly-fetched route. None on disabled / failure."""
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
    _log_done(key, "fresh" if fresh else "missing", started)
    return fresh


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
    try:
        async with _http_client() as client:
            resp = await client.get(
                _DIRECTIONS_URL,
                params={
                    "origin": f"{from_lat},{from_lng}",
                    "destination": f"{to_lat},{to_lng}",
                    "mode": mode,
                    "key": settings.google_api_key,
                },
            )
            if resp.status_code == 429:
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
