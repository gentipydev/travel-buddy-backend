"""Routed polylines via the Mapbox Directions API, with a coords+mode cache.

Cache rounds lat/lng to 4 decimals (~10m precision) so trips that share
endpoints — "near Berat castle" → "Berat town center" — reuse the same
polyline regardless of micro-differences in geocoding.

`geometries=polyline` returns a precision-5 encoded polyline — the encoding
`DirectionsCache.encoded_polyline`, the static-map downsampler and the Flutter
`polyline_decoder.dart` already speak, so nothing downstream changes.

There is no snap-to-roads retry step: Mapbox snaps each waypoint to the road
network itself (see `radiuses`), so a `NoSegment` response means the stop
genuinely isn't near a routable road.
"""
from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import DirectionsCache
from app.integrations._retry import http_get_with_retry

logger = logging.getLogger(__name__)


_DIRECTIONS_URL = "https://api.mapbox.com/directions/v5/mapbox"
_HTTP_TIMEOUT = 5.0
_CACHE_TTL = timedelta(days=90)
# How far Mapbox may move a waypoint to reach the road network. Generous
# enough for a trailhead or a beach pin, tight enough that a polyline never
# visibly starts in a different part of town.
_SNAP_RADIUS_M = 250


# Module-level singleton — same pattern as `mapbox_geocoding.py`. Keeps
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

    Failures are never fatal: the caller omits the leg and the client draws a
    straight connector between the two stops instead.
    """
    started = time.perf_counter()
    key = _cache_key(from_lat, from_lng, to_lat, to_lng, mode)

    if not settings.mapbox_access_token or not settings.mapbox_directions_enabled:
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
    now = datetime.now(UTC)
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
    # Mapbox takes coordinates longitude-first, semicolon-separated.
    coords = f"{from_lng},{from_lat};{to_lng},{to_lat}"
    try:
        resp = await http_get_with_retry(
            client,
            f"{_DIRECTIONS_URL}/{mode}/{coords}",
            params={
                "geometries": "polyline",
                "overview": "full",
                "radiuses": f"{_SNAP_RADIUS_M};{_SNAP_RADIUS_M}",
                "access_token": settings.mapbox_access_token,
            },
        )
        if resp.status_code == 429:
            # Reached only when retries are exhausted (policy tries 429
            # up to 3 times). Treat as quota_exceeded for visibility.
            logger.warning(
                "mapbox_api=directions_get cache_key=%s status=quota_exceeded",
                key,
            )
            return None
        if resp.status_code != 200:
            logger.warning(
                "mapbox_api=directions_get cache_key=%s status=%s detail=%s",
                key,
                resp.status_code,
                _error_message(resp),
            )
            return None
        data = resp.json()
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            "mapbox_api=directions_get cache_key=%s error=%s",
            key,
            type(exc).__name__,
        )
        return None
    except Exception:
        logger.exception(
            "mapbox_api=directions_get cache_key=%s unexpected", key
        )
        return None

    # `code` is "Ok" on success; "NoRoute" (no path exists, e.g. walking
    # across a bay) and "NoSegment" (a waypoint is beyond `radiuses` of any
    # road) both come back 200. Don't cache either — the orchestrator falls
    # back to a straight line client-side.
    code = data.get("code")
    routes = data.get("routes") or []
    if code != "Ok" or not routes:
        logger.info(
            "mapbox_api=directions_get cache_key=%s api_code=%s", key, code
        )
        return None

    route = routes[0]
    geometry = route.get("geometry")
    if not geometry:
        return None
    # Mapbox reports metres / seconds as floats; the columns are integers.
    distance = route.get("distance")
    duration = route.get("duration")

    return await _upsert_cache(
        db,
        key=key,
        encoded_polyline=geometry,
        distance_meters=round(distance) if distance is not None else None,
        duration_seconds=round(duration) if duration is not None else None,
        travel_mode=mode,
    )


def _error_message(resp: httpx.Response) -> str:
    """Best-effort extraction of the Mapbox error message.

    Error bodies look like `{"message": "Not Authorized - Invalid Token"}`.
    Falls back to a truncated body.
    """
    try:
        return (resp.json() or {}).get("message") or resp.text[:200]
    except Exception:
        return resp.text[:200]


async def _upsert_cache(
    db: AsyncSession,
    *,
    key: str,
    encoded_polyline: str,
    distance_meters: int | None,
    duration_seconds: int | None,
    travel_mode: str,
) -> DirectionsCache:
    now = datetime.now(UTC)
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
        "mapbox_api=directions_get cache_key=%s source=%s latency_ms=%s",
        key,
        source,
        elapsed_ms,
    )
