"""Resolve a stop's name (+ city) to coordinates and a formatted address.

Three tiers, in order:
    1. Curated `places` table (vetted Albanian destinations) — local-only,
       always wins. Kill switch does not gate this tier, and it is the only
       tier that can supply photos or a rating.
    2. `geocode_cache` table — Mapbox responses kept for 90 days, keyed by the
       normalized "{name}|{city}" query.
    3. Mapbox Search Box forward search — the miss path; result is cached.

Saves never fail because of this module. Every error path logs and returns
None so the caller can proceed without enrichment. The `source` field on
`GeocodeResult` is the metric for both curated coverage and cache-hit rate.

Photos and ratings are curated-only by design: Mapbox returns neither, and
the client renders its own placeholder for stops without a photo.
"""
from __future__ import annotations

import logging
import time
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx
from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.db.models import GeocodeCache, Place
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


# Search Box forward search. Unlike `/suggest` + `/retrieve`, `/forward` is a
# single stateless request (no session token) and returns coordinates plus a
# formatted address in one hop — so there is no separate "details" call.
_FORWARD_URL = "https://api.mapbox.com/search/searchbox/v1/forward"
# Albania's bounding box, minLon,minLat,maxLon,maxLat. A hard server-side
# filter on top of `country=al`; `_is_in_albania` re-checks the result because
# bbox is a bias for some feature types rather than a guarantee.
_ALBANIA_BBOX = "19.26,39.63,21.06,42.67"
# Administrative feature types. When forward search can't match the POI it was
# asked for, it does NOT return empty — it falls back to the surrounding
# context and hands back the city. "Qwertyuiop Asdfgh Restaurant, Tirana" comes
# back as feature_type=place, name="Tirana". Accepting that would drop a marker
# in the middle of Tirana for a stop the model invented.
# See `_is_context_fallback` for how that is told apart from a real hit.
_COARSE_FEATURE_TYPES = frozenset(
    {"place", "region", "district", "country", "postcode", "locality"}
)
_HTTP_TIMEOUT = 5.0
_CACHE_TTL = timedelta(days=90)
_MAX_PHOTOS = 5   # curated rows only; the caller caps at 3 for the UI


@dataclass
class GeocodeResult:
    lat: float
    lng: float
    formatted_address: str | None
    # Curated tiers only. Mapbox returns neither, so a stop resolved from
    # cache or the API carries `rating=None` and `photos=[]`, and the client
    # falls back to its place placeholder.
    rating: float | None = None
    photos: list[str] = field(default_factory=list)
    source: Literal["curated", "cache", "fresh"] = "fresh"


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


def _normalize(text: str | None) -> str:
    """Casefold + strip accents so "Gjirokastër" and "gjirokaster" collide.

    Used for both the curated-table match and the cache key, so the two tiers
    agree on what counts as the same place.
    """
    if not text:
        return ""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


def cache_key(name: str, city: str | None) -> str:
    return f"{_normalize(name)}|{_normalize(city)}"


def _error_message(resp: httpx.Response) -> str:
    """Best-effort extraction of the Mapbox error message.

    Error bodies look like `{"message": "Not Authorized - Invalid Token"}`.
    Falls back to a truncated body.
    """
    try:
        return (resp.json() or {}).get("message") or resp.text[:200]
    except Exception:
        return resp.text[:200]


async def resolve_place(
    db: AsyncSession,
    *,
    name: str,
    city: str | None = None,
) -> GeocodeResult | None:
    """Curated → cache → Mapbox. Returns None only when all three miss."""
    started = time.perf_counter()
    key = cache_key(name, city)

    # Tier 1: curated.
    curated = await _read_curated(db, name=name, city=city)
    if curated is not None:
        _log_done(key, "curated", started)
        return curated

    # Kill switch only gates tiers 2-3. Curated is local-only.
    if not settings.mapbox_access_token or not settings.mapbox_geocoding_enabled:
        _log_done(key, "missing", started)
        return None

    # Tier 2: cache.
    cached = await _read_cache(db, key)
    if cached is not None:
        _log_done(key, "cache", started)
        return cached

    # Tier 3: Mapbox forward search.
    fresh = await _fetch_and_cache(db, key=key, name=name, city=city)
    _log_done(key, "fresh" if fresh else "missing", started)
    return fresh


async def _read_curated(
    db: AsyncSession, *, name: str, city: str | None
) -> GeocodeResult | None:
    """Exact (accent- and case-insensitive) name match in the curated table.

    Matching is on name because the curated rows carry no external place id.
    When several rows share a name, a matching `city` breaks the tie; failing
    that the highest-rated row wins, mirroring `rag.place_detection`.
    """
    normalized = _normalize(name)
    if not normalized:
        return None
    rows = (
        await db.execute(
            select(Place).where(func.lower(Place.name) == name.strip().lower())
        )
    ).scalars().all()
    # The SQL comparison is case-insensitive but not accent-insensitive, so
    # re-filter in Python against the same normalization the cache key uses.
    candidates = [p for p in rows if _normalize(p.name) == normalized]
    if not candidates:
        return None

    wanted_city = _normalize(city)
    if wanted_city:
        by_city = [p for p in candidates if _normalize(p.city) == wanted_city]
        if by_city:
            candidates = by_city
    place = max(candidates, key=lambda p: p.rating or 0)

    if place.latitude is None or place.longitude is None:
        # A curated row with no coords can't place a marker; fall through to
        # the geocoder rather than returning a half-result.
        return None

    photos = list(place.photos or [])[:_MAX_PHOTOS]
    address = f"{place.name}, {place.city}" if place.city else place.name
    return GeocodeResult(
        lat=place.latitude,
        lng=place.longitude,
        formatted_address=address,
        rating=place.rating,
        photos=photos,
        source="curated",
    )


async def _read_cache(db: AsyncSession, key: str) -> GeocodeResult | None:
    now = datetime.now(UTC)
    result = await db.execute(
        select(GeocodeCache).where(
            GeocodeCache.query_key == key,
            GeocodeCache.expires_at > now,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None
    return GeocodeResult(
        lat=row.lat,
        lng=row.lng,
        formatted_address=row.formatted_address,
        source="cache",
    )


async def _fetch_and_cache(
    db: AsyncSession,
    *,
    key: str,
    name: str,
    city: str | None,
) -> GeocodeResult | None:
    # Always append ", Albania" so the model's city alone can't drift to a
    # same-named locality elsewhere. `country=al` + `bbox` are the server-side
    # filters; the `_is_in_albania` check below is the hard guard.
    query = f"{name}, {city}, Albania" if city else f"{name}, Albania"
    params: dict[str, str | int] = {
        "q": query,
        "country": "al",
        "bbox": _ALBANIA_BBOX,
        "limit": 1,
        "language": "en",
        "access_token": settings.mapbox_access_token,
    }
    if settings.mapbox_permanent_geocoding:
        # Mapbox only licenses storing geocoding results under `permanent`.
        # We cache and persist coordinates either way, so this must be on
        # (and enabled on the account) before any real launch.
        params["permanent"] = "true"

    client = _http_client()
    try:
        resp = await http_get_with_retry(client, _FORWARD_URL, params=params)
        if resp.status_code == 429:
            # Reached only when retries are exhausted — the retry policy
            # tries 429 up to 3 times. Treat as quota_exceeded for
            # operator visibility.
            logger.warning(
                "mapbox_api=geocode query=%r status=quota_exceeded", query
            )
            return None
        if resp.status_code != 200:
            logger.warning(
                "mapbox_api=geocode query=%r status=%s detail=%s",
                query,
                resp.status_code,
                _error_message(resp),
            )
            return None
        data = resp.json() or {}
        features = data.get("features") or []
    except (httpx.RequestError, httpx.TimeoutException) as exc:
        logger.warning(
            "mapbox_api=geocode query=%r error=%s", query, type(exc).__name__
        )
        return None
    except Exception:
        logger.exception("mapbox_api=geocode query=%r unexpected", query)
        return None

    if not features:
        return None
    props = features[0].get("properties") or {}
    lat, lng = _coords_from(features[0])
    if lat is None or lng is None:
        return None

    if not _is_in_albania(lat, lng):
        # Hard guard against the unbounded-search failure: refuse to return
        # coords outside the Albania bbox, and refuse to cache them — once
        # cached, every later caller for the same name would get the wrong
        # country. The caller leaves the stop un-geo'd, which is correct
        # fail-safe behavior.
        logger.warning(
            "mapbox_api=geocode query=%r dropped=out_of_bounds lat=%s lng=%s",
            query, lat, lng,
        )
        return None

    if _is_context_fallback(props, name, city):
        # Not a match — forward search fell back to the enclosing city. Return
        # None (and don't cache) so the stop stays un-geo'd rather than showing
        # a marker in the wrong place.
        logger.info(
            "mapbox_api=geocode query=%r dropped=context_fallback resolved=%r",
            query, props.get("name"),
        )
        return None

    formatted = props.get("full_address") or props.get("place_formatted")
    resolved_name = props.get("name") or name

    await _upsert_cache(
        db,
        key=key,
        name=resolved_name,
        formatted_address=formatted,
        lat=lat,
        lng=lng,
        raw_response=features[0],
    )

    return GeocodeResult(
        lat=lat,
        lng=lng,
        formatted_address=formatted,
        source="fresh",
    )


def _is_context_fallback(
    props: dict, requested_name: str, requested_city: str | None
) -> bool:
    """True when the result is the enclosing city rather than the place asked for.

    Forward search never returns an empty feature list for a query that carries
    a real city — it degrades to that city. The tell is a coarse feature type
    whose name is the *city* we passed while the stop we asked for is something
    else: "Qwertyuiop Asdfgh Restaurant, Tirana" → place "Tirana".

    Comparing against the city rather than against the stop name matters,
    because a correct coarse hit routinely comes back under a different label
    than the one queried — Albanian place names carry definite/indefinite forms
    and English suffixes that no normalization will reconcile:

        "Sarandë", city "Vlorë"  → place "Saranda City"   (correct)
        "Kruja",   city "Durrës" → place "Krujë"          (correct)

    Neither of those names is the city, so both are kept. A stop that genuinely
    *is* its city ("Berat" in Berat, a whole-day stop) also survives, because
    the requested name matches the city too.
    """
    if props.get("feature_type") not in _COARSE_FEATURE_TYPES:
        return False
    city = _normalize(requested_city)
    if not city:
        # Nothing to have fallen back to — the query carried no city context.
        return False
    return _normalize(props.get("name")) == city and _normalize(requested_name) != city


def _coords_from(feature: dict) -> tuple[float | None, float | None]:
    """Pull (lat, lng) out of a Search Box feature.

    `properties.coordinates` is the documented shape; `geometry.coordinates`
    ([lng, lat] — GeoJSON order) is the fallback for feature types that omit
    it. Returns (None, None) when neither is usable.
    """
    props = feature.get("properties") or {}
    coords = props.get("coordinates") or {}
    lat = coords.get("latitude")
    lng = coords.get("longitude")
    if lat is not None and lng is not None:
        return float(lat), float(lng)

    geometry = (feature.get("geometry") or {}).get("coordinates") or []
    if len(geometry) >= 2:
        return float(geometry[1]), float(geometry[0])
    return None, None


async def _upsert_cache(
    db: AsyncSession,
    *,
    key: str,
    name: str,
    formatted_address: str | None,
    lat: float,
    lng: float,
    raw_response: dict,
) -> None:
    now = datetime.now(UTC)
    expires = now + _CACHE_TTL
    stmt = (
        pg_insert(GeocodeCache)
        .values(
            query_key=key,
            name=name,
            formatted_address=formatted_address,
            lat=lat,
            lng=lng,
            raw_response=raw_response,
            fetched_at=now,
            expires_at=expires,
        )
        .on_conflict_do_update(
            index_elements=["query_key"],
            set_={
                "name": name,
                "formatted_address": formatted_address,
                "lat": lat,
                "lng": lng,
                "raw_response": raw_response,
                "fetched_at": now,
                "expires_at": expires,
            },
        )
    )
    await db.execute(stmt)


def _log_done(key: str, source: str, started: float) -> None:
    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "mapbox_api=geocode key=%r source=%s latency_ms=%s",
        key,
        source,
        elapsed_ms,
    )
