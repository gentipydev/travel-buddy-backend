"""Best-effort enrichment of a saved trip.

Runs after every save / refine (via `serialize_trip` in
`app/domain/trips/repository.py`). Never raises: a failure in any phase
is logged and the trip ends up with whatever data we managed to gather.
Saves never block on this module.

Phases:
    1. Load trip + days + stops + existing routes; truncate routes (refines
       produce a clean route set).
    2. Place resolution in parallel — curated/cache/Mapbox. Backfills each
       stop's lat/lng and formatted_address; photo_url/photos/rating come
       from the curated tier only (Mapbox supplies neither, and the client
       renders its own placeholder for stops without a photo).
    3. Directions in parallel for consecutive stops with coords. Inserts
       trip_routes rows; failed segments are skipped (the client draws a
       straight line as fallback).
    4. Build a Mapbox Static Images URL covering all geo'd stops + the new
       polylines; persist on Trip.static_map_url.
    5. Commit.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Mapping, Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.db.models import Trip, TripDay, TripRoute, TripStop
from app.domain.trips.geo import haversine_km
from app.integrations.mapbox_directions import get_route
from app.integrations.mapbox_geocoding import GeocodeResult, resolve_place
from app.integrations.mapbox_static import build_static_map_url


logger = logging.getLogger(__name__)


_WALKING_THRESHOLD_KM = 2.0


async def enrich_trip(db: AsyncSession, trip_id: UUID) -> None:
    """Best-effort enrichment of a saved trip. Never raises."""
    started = time.perf_counter()
    place_sources: Counter[str] = Counter()
    routes_created = 0
    routes_attempted = 0
    static_map_status = "skipped"

    # Phase 1 — load.
    trip = await _load(db, trip_id)
    if trip is None:
        logger.warning("enrich_trip trip_id=%s missing", trip_id)
        return

    # Wipe the previous route set so refines (reorders, swaps, deletes) end
    # up with exactly the routes that match current geometry.
    await db.execute(delete(TripRoute).where(TripRoute.trip_id == trip_id))

    flat_stops: list[TripStop] = [
        stop for day in trip.days for stop in day.stops
    ]

    # Phase 2 — places (parallel).
    try:
        place_sources = await _enrich_places(db, flat_stops)
    except Exception:
        logger.exception("enrich_trip phase=places trip_id=%s", trip_id)

    # Phase 3 — directions (parallel; depends on phase-2 lat/lng backfill).
    try:
        routes_attempted, routes_created = await _create_routes(
            db, trip_id, trip.days
        )
    except Exception:
        logger.exception("enrich_trip phase=directions trip_id=%s", trip_id)

    # Phase 4 — static map URL. Re-read trip_routes so the builder sees
    # exactly what we just inserted (including any failures that left gaps).
    url: str | None = None
    try:
        new_routes = (
            await db.execute(
                select(TripRoute).where(TripRoute.trip_id == trip_id)
            )
        ).scalars().all()
        url = _build_static_map(trip.days, new_routes)
        static_map_status = "ok" if url else "empty"
    except Exception:
        logger.exception("enrich_trip phase=static_map trip_id=%s", trip_id)
        static_map_status = "error"

    if url is not None:
        trip.static_map_url = url

    # Phase 5 — commit.
    try:
        await db.commit()
    except Exception:
        logger.exception("enrich_trip phase=commit trip_id=%s", trip_id)
        await db.rollback()

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    logger.info(
        "enrich_trip trip_id=%s places=%s/%s "
        "(curated=%s cache=%s fresh=%s missing=%s) "
        "routes=%s/%s static_map=%s elapsed_ms=%s",
        trip_id,
        sum(v for k, v in place_sources.items() if k != "missing"),
        sum(place_sources.values()),
        place_sources.get("curated", 0),
        place_sources.get("cache", 0),
        place_sources.get("fresh", 0),
        place_sources.get("missing", 0),
        routes_created,
        routes_attempted,
        static_map_status,
        elapsed_ms,
    )


async def _load(db: AsyncSession, trip_id: UUID) -> Trip | None:
    result = await db.execute(
        select(Trip)
        .where(Trip.id == trip_id)
        .options(
            selectinload(Trip.days).selectinload(TripDay.stops),
            selectinload(Trip.routes),
        )
    )
    return result.unique().scalar_one_or_none()


async def _enrich_places(
    db: AsyncSession, stops: Sequence[TripStop]
) -> Counter[str]:
    """Resolve stops in parallel; mutate each stop in-place. Returns a
    counter with `curated`/`cache`/`fresh`/`missing` for the log line.

    Single-phase: Mapbox forward search returns coordinates and a formatted
    address in one request, so there is no separate id-lookup step. Stops
    without a usable name can't be resolved at all and count as missing.
    """
    counter: Counter[str] = Counter()

    targets = [s for s in stops if (s.name or "").strip()]
    counter["missing"] = len(stops) - len(targets)
    if not targets:
        return counter

    results = await asyncio.gather(
        *[
            resolve_place(db, name=(s.name or "").strip(), city=s.city)
            for s in targets
        ],
        return_exceptions=True,
    )
    for stop, result in zip(targets, results):
        if isinstance(result, Exception):
            logger.warning(
                "resolve_place name=%r city=%r raised=%s",
                stop.name,
                stop.city,
                type(result).__name__,
            )
            counter["missing"] += 1
            continue
        if result is None:
            counter["missing"] += 1
            continue
        _apply_to_stop(stop, result)
        counter[result.source] += 1
    return counter


def _apply_to_stop(stop: TripStop, result: GeocodeResult) -> None:
    # Photos and rating are curated-only — Mapbox has neither. Always write
    # `photos` so the list stays in sync with what we just resolved (a
    # re-enrichment that finds no curated row shouldn't leave stale entries
    # from a previous run). Cap at 3 — that's all the UI uses.
    stop.photos = result.photos[:3]
    stop.photo_url = stop.photos[0] if stop.photos else None
    stop.rating = result.rating
    if result.formatted_address is not None:
        stop.formatted_address = result.formatted_address
    # Backfill coords only when missing — Claude's value wins if both have one.
    if stop.lat is None:
        stop.lat = result.lat
    if stop.lng is None:
        stop.lng = result.lng


def _route_pairs(
    days: Sequence[TripDay],
) -> list[tuple[TripDay, TripStop, TripStop]]:
    """Consecutive (lat,lng)-bearing stop pairs to route: every adjacent pair
    within a day, plus each consecutive-day transition (last geo'd stop of
    day N → first geo'd stop of day N+1). Cross-day pairs are owned by the
    next day so their trip_day_id matches the day the traveler heads INTO.
    Shared by the persisted (`_create_routes`) and preview route builders.
    """
    pairs: list[tuple[TripDay, TripStop, TripStop]] = []
    for day in days:
        ordered = sorted(day.stops, key=lambda s: s.sort_order)
        for prev, nxt in zip(ordered, ordered[1:]):
            if (
                prev.lat is not None and prev.lng is not None
                and nxt.lat is not None and nxt.lng is not None
            ):
                pairs.append((day, prev, nxt))

    for prev_day, next_day in zip(days, days[1:]):
        prev_ordered = sorted(prev_day.stops, key=lambda s: s.sort_order)
        next_ordered = sorted(next_day.stops, key=lambda s: s.sort_order)
        last = next(
            (
                s for s in reversed(prev_ordered)
                if s.lat is not None and s.lng is not None
            ),
            None,
        )
        first = next(
            (
                s for s in next_ordered
                if s.lat is not None and s.lng is not None
            ),
            None,
        )
        if last is not None and first is not None:
            pairs.append((next_day, last, first))

    return pairs


def _route_mode(prev: TripStop, nxt: TripStop) -> str:
    return (
        "walking"
        if haversine_km(prev.lat, prev.lng, nxt.lat, nxt.lng)
        < _WALKING_THRESHOLD_KM
        else "driving"
    )


async def _warm_connection(db: AsyncSession) -> None:
    """Establish the session's DB connection before a Directions fan-out.

    `get_route` calls run via `asyncio.gather` on this one shared session, and
    AsyncSession forbids concurrent operations — when several legs miss the
    DirectionsCache at once on a *cold* session, they race connection
    provisioning and one raises `InvalidRequestError: concurrent operations
    are not permitted`. Forcing the connection open first serialises that
    provisioning so the cache reads/writes don't collide. Best-effort: a
    failure here just falls through to the gather (cache hits still work).
    """
    try:
        await db.connection()
    except Exception:
        logger.debug("warm_connection failed", exc_info=True)


async def _create_routes(
    db: AsyncSession,
    trip_id: UUID,
    days: Sequence[TripDay],
) -> tuple[int, int]:
    """Schedule one Directions call per routable stop pair (see `_route_pairs`)
    and persist a `TripRoute` per success. Returns (attempted, created).
    """
    pairs = _route_pairs(days)
    if not pairs:
        return 0, 0

    await _warm_connection(db)
    tasks = [
        get_route(db, prev.lat, prev.lng, nxt.lat, nxt.lng, _route_mode(prev, nxt))
        for _, prev, nxt in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    created = 0
    for (day, prev, nxt), result in zip(pairs, results):
        # DEBUG(polyline): per-leg outcome so we can see exactly which stop
        # pairs got a routed polyline vs. fell through to a straight line.
        if isinstance(result, Exception):
            logger.warning(
                "create_routes leg %s -> %s raised=%s",
                prev.name, nxt.name, type(result).__name__,
            )
            continue
        if result is None:
            logger.info(
                "create_routes leg %s -> %s result=none (straight line)",
                prev.name, nxt.name,
            )
            continue
        logger.info(
            "create_routes leg %s -> %s mode=%s dist_m=%s poly_len=%s",
            prev.name, nxt.name, result.travel_mode,
            result.distance_meters, len(result.encoded_polyline),
        )
        db.add(
            TripRoute(
                trip_id=trip_id,
                trip_day_id=day.id,
                from_stop_id=prev.id,
                to_stop_id=nxt.id,
                encoded_polyline=result.encoded_polyline,
                distance_meters=result.distance_meters,
                duration_seconds=result.duration_seconds,
                travel_mode=result.travel_mode,
            )
        )
        created += 1
    if created:
        await db.flush()
    return len(pairs), created


async def _create_preview_routes(
    db: AsyncSession,
    days: Sequence[TripDay],
) -> list[TripRoute]:
    """Build road-following routes for the live preview WITHOUT persisting.

    Same pairing + Directions fan-out as `_create_routes`, but returns
    transient `TripRoute` objects (never `db.add`-ed) so they ride along on
    the in-memory `trip.routes` for serialization only. The preview trip has
    no DB row, so persisting these would violate the trip/stop foreign keys.
    `get_route` still reads/writes the DirectionsCache table (shared, real),
    so a later Save finds the legs already cached. Best-effort; failures fall
    back to the client's straight-line connector, same as a saved trip.
    """
    pairs = _route_pairs(days)
    if not pairs:
        return []

    await _warm_connection(db)
    tasks = [
        get_route(db, prev.lat, prev.lng, nxt.lat, nxt.lng, _route_mode(prev, nxt))
        for _, prev, nxt in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    routes: list[TripRoute] = []
    for (day, prev, nxt), result in zip(pairs, results):
        if isinstance(result, Exception) or result is None:
            logger.info(
                "preview_routes leg %s -> %s result=none (straight line)",
                prev.name, nxt.name,
            )
            continue
        routes.append(
            TripRoute(
                id=uuid4(),
                trip_id=None,
                trip_day_id=day.id,
                from_stop_id=prev.id,
                to_stop_id=nxt.id,
                encoded_polyline=result.encoded_polyline,
                distance_meters=result.distance_meters,
                duration_seconds=result.duration_seconds,
                travel_mode=result.travel_mode,
            )
        )
    logger.info(
        "preview_routes attempted=%s created=%s", len(pairs), len(routes)
    )
    return routes


def _build_static_map(
    days: Sequence[TripDay], routes: Sequence[TripRoute]
) -> str | None:
    markers_by_day: list[list[tuple[float, float]]] = []
    for day in days:
        ordered = sorted(day.stops, key=lambda s: s.sort_order)
        markers_by_day.append(
            [(s.lat, s.lng) for s in ordered if s.lat is not None and s.lng is not None]
        )
    # Drop trailing empty-marker days so we don't waste color slots on them.
    if not any(markers_by_day):
        return None
    polylines = [r.encoded_polyline for r in routes]

    if not settings.mapbox_access_token:
        return None

    return build_static_map_url(
        access_token=settings.mapbox_access_token,
        markers_by_day=markers_by_day,
        polylines=polylines,
    )


async def geocode_stop_preview(
    db: AsyncSession,
    *,
    name: str,
    city: str | None,
    lat: float | None,
    lng: float | None,
) -> tuple[float, float] | None:
    """Resolve one stop's coordinates for the live map, no persistence.

    Used by `/chat/stream` to drop a marker the instant a stop streams out
    of the itinerary tool call. Prefers coords Claude already supplied;
    otherwise runs the same `resolve_place` lookup every saved trip uses
    (which may write the shared geocode cache, but touches nothing else).
    Returns `(lat, lng)` or None when the stop can't be mapped. Never raises.
    """
    if lat is not None and lng is not None:
        return lat, lng
    if not (name or "").strip():
        return None
    try:
        result = await resolve_place(db, name=name, city=city)
    except Exception:
        logger.warning("geocode_stop_preview name=%r raised", name, exc_info=True)
        return None
    if result is None:
        return None
    return result.lat, result.lng


async def build_live_route_dicts(
    db: AsyncSession,
    stops: Sequence[Mapping[str, object]],
) -> list[dict]:
    """Road-following routes for the per-stop *live-streamed* preview.

    `stops` is the ordered list of stops already pushed to the client as
    `itinerary_stop` frames — each a mapping with `id` (the `live-{day}-{sort}`
    id the client keyed its marker on), `day_number`, `lat`, `lng`. Adjacent
    pairs in this ordered list cover both intra-day legs and day-to-day
    transitions (last stop of day N → first of day N+1), exactly matching
    `_route_pairs`. Returns serializable route dicts whose `from_stop_id` /
    `to_stop_id` match the streamed marker ids, so the client merges them onto
    the existing live trip with no marker re-animation. No DB writes beyond the
    shared DirectionsCache; best-effort (failed legs omitted → straight line).
    """
    routable = [
        s for s in stops if s.get("lat") is not None and s.get("lng") is not None
    ]
    if len(routable) < 2:
        return []

    pairs = list(zip(routable, routable[1:]))
    await _warm_connection(db)
    tasks = [
        get_route(
            db,
            float(prev["lat"]), float(prev["lng"]),
            float(nxt["lat"]), float(nxt["lng"]),
            (
                "walking"
                if haversine_km(
                    float(prev["lat"]), float(prev["lng"]),
                    float(nxt["lat"]), float(nxt["lng"]),
                ) < _WALKING_THRESHOLD_KM
                else "driving"
            ),
        )
        for prev, nxt in pairs
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    routes: list[dict] = []
    for (prev, nxt), result in zip(pairs, results):
        if isinstance(result, Exception) or result is None:
            logger.info(
                "live_routes leg %s -> %s result=none (straight line)",
                prev.get("id"), nxt.get("id"),
            )
            continue
        routes.append(
            {
                "id": str(uuid4()),
                # Mirror the client's live day id (appendLiveStop builds
                # `live-day-{n}`) so the route reads as owned by that day.
                "trip_day_id": f"live-day-{nxt.get('day_number')}",
                "from_stop_id": prev["id"],
                "to_stop_id": nxt["id"],
                "encoded_polyline": result.encoded_polyline,
                "distance_meters": result.distance_meters,
                "duration_seconds": result.duration_seconds,
                "travel_mode": result.travel_mode,
            }
        )
    logger.info(
        "live_routes attempted=%s created=%s", len(pairs), len(routes)
    )
    return routes


async def geocode_itinerary_preview(db: AsyncSession, payload: dict) -> dict:
    """Geocode a draft itinerary WITHOUT persisting it, for the live map.

    Used by `/chat/stream` to light up the map the moment Claude emits an
    itinerary, before (and independent of) any "Save". Builds transient ORM
    objects from the validated `emit_itinerary` payload, runs the same
    `_enrich_places` resolution every saved trip goes through (curated row →
    geocode cache → Mapbox forward search, for lat/lng + address), and
    serializes the result into the canonical `TripOut` shape.

    Two deliberate departures from `enrich_trip`:
      - Nothing touches the session/DB beyond cache reads — no trip row, so
        no impact on the per-user trip limit, and anonymous users work too.
      - Directions are skipped (`routes: []`). The client falls back to a
        straight dashed connector between stops; the road-following polyline
        and static map are computed on Save. Most previews are never saved,
        so firing Directions per draft would be wasted spend.

    Never raises on enrichment failure internally (mirrors `_enrich_places`,
    which logs and continues); the caller wraps this best-effort regardless.
    """
    now = datetime.now(timezone.utc)
    trip = Trip(
        id=uuid4(),
        title=payload.get("title") or "Trip",
        summary=payload.get("summary"),
        static_map_url=None,
        created_at=now,
        updated_at=now,
    )
    # start_date arrives as an ISO date string in the wire payload; keep the
    # _serialize contract (it calls `.isoformat()`), so parse to a date.
    start_date = payload.get("start_date")
    if isinstance(start_date, str) and start_date:
        try:
            trip.start_date = datetime.fromisoformat(start_date).date()
        except ValueError:
            trip.start_date = None
    else:
        trip.start_date = None

    days: list[TripDay] = []
    for day_payload in payload.get("days", []):
        day = TripDay(
            id=uuid4(),
            day_number=day_payload.get("day_number"),
            title=day_payload.get("title"),
            base_city=day_payload.get("base_city"),
        )
        stops: list[TripStop] = []
        for index, stop_payload in enumerate(day_payload.get("stops", [])):
            stops.append(
                TripStop(
                    id=uuid4(),
                    sort_order=index,
                    name=stop_payload.get("name"),
                    kind=stop_payload.get("kind"),
                    city=stop_payload.get("city"),
                    address=stop_payload.get("address"),
                    duration_minutes=stop_payload.get("duration_minutes"),
                    arrival_time=stop_payload.get("arrival_time"),
                    end_time=stop_payload.get("end_time"),
                    transportation=stop_payload.get("transportation"),
                    notes=stop_payload.get("notes"),
                    lat=stop_payload.get("lat"),
                    lng=stop_payload.get("lng"),
                    photos=[],
                )
            )
        day.stops = stops
        days.append(day)
    trip.days = days
    trip.routes = []

    flat_stops = [stop for day in trip.days for stop in day.stops]
    if flat_stops:
        try:
            await _enrich_places(db, flat_stops)
        except Exception:
            logger.exception("geocode_itinerary_preview phase=places")

    # Route the preview so the live map draws road-following polylines (not
    # straight connectors) the moment the itinerary streams in. Transient
    # routes only — nothing persisted; see `_create_preview_routes`. Best-
    # effort: any failure leaves routes=[] and the client falls back to
    # straight lines, exactly as before.
    try:
        trip.routes = await _create_preview_routes(db, trip.days)
    except Exception:
        logger.exception("geocode_itinerary_preview phase=directions")
        trip.routes = []

    # Local import dodges the repository ↔ enrichment cycle (repository
    # already imports this module lazily).
    from app.domain.trips.repository import _serialize

    return _serialize(trip)
