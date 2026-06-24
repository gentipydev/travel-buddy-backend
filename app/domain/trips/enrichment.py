"""Best-effort enrichment of a saved trip.

Runs after every save / refine (via `serialize_trip` in
`app/domain/trips/repository.py`). Never raises: a failure in any phase
is logged and the trip ends up with whatever data we managed to gather.
Saves never block on this module.

Phases:
    1. Load trip + days + stops + existing routes; truncate routes (refines
       produce a clean route set).
    2. Places enrichment in parallel — curated/cache/Google. Updates each
       stop's photo_url/rating/formatted_address; backfills lat/lng when
       Claude's output was sparse but the curated row has them.
    3. Directions in parallel for consecutive stops with coords. Inserts
       trip_routes rows; failed segments are skipped (the client draws a
       straight line as fallback).
    4. Build signed Static Maps URL covering all geo'd stops + the new
       polylines; persist on Trip.static_map_url.
    5. Commit.
"""
from __future__ import annotations

import asyncio
import logging
import time
from collections import Counter
from datetime import datetime, timezone
from typing import Sequence
from uuid import UUID, uuid4

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.config import settings
from app.db.models import Trip, TripDay, TripRoute, TripStop
from app.domain.trips.geo import haversine_km
from app.integrations.google_directions import get_route
from app.integrations.google_places import (
    EnrichmentResult,
    enrich_stop,
    find_place_id_by_name,
)
from app.integrations.static_maps import build_static_map_url


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

    Two-phase. Stops without a `google_place_id` first go through Google
    Places Text Search to resolve one from name+city — Claude tends to
    produce names but no IDs, and without an ID there's nothing to enrich
    or geocode. Resolved stops then merge back with stops that already
    had an ID and run the full curated/cache/fresh path.
    """
    counter: Counter[str] = Counter()

    # Phase A — text-search fallback for stops missing google_place_id.
    needs_resolution = [
        s for s in stops if not s.google_place_id and (s.name or "").strip()
    ]
    if needs_resolution:
        await _resolve_missing_place_ids(needs_resolution)

    # Phase B — enrich stops that have an ID (original or just resolved).
    targets = [s for s in stops if s.google_place_id]
    counter["missing"] = len(stops) - len(targets)
    if not targets:
        return counter

    results = await asyncio.gather(
        *[enrich_stop(db, s.google_place_id) for s in targets],
        return_exceptions=True,
    )
    for stop, result in zip(targets, results):
        if isinstance(result, Exception):
            logger.warning(
                "enrich_stop place_id=%s raised=%s",
                stop.google_place_id,
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


async def _resolve_missing_place_ids(stops: Sequence[TripStop]) -> None:
    """Run Google Places Text Search in parallel for each stop and write
    `google_place_id` (and backfill lat/lng when missing) when a result
    comes back. Stops where the search returns nothing are left alone —
    the caller counts them as missing in the next phase."""
    queries = [(s, (s.name or "").strip(), s.city) for s in stops]
    results = await asyncio.gather(
        *[find_place_id_by_name(name=name, city=city) for _, name, city in queries],
        return_exceptions=True,
    )
    resolved = 0
    for (stop, _name, _city), result in zip(queries, results):
        if isinstance(result, Exception):
            logger.warning(
                "find_place_id name=%r city=%r raised=%s",
                stop.name,
                stop.city,
                type(result).__name__,
            )
            continue
        if result is None:
            continue
        place_id, lat, lng = result
        stop.google_place_id = place_id
        if stop.lat is None:
            stop.lat = lat
        if stop.lng is None:
            stop.lng = lng
        resolved += 1
    logger.info(
        "trip_enrichment textsearch_resolved=%d/%d", resolved, len(stops)
    )


def _apply_to_stop(stop: TripStop, result: EnrichmentResult) -> None:
    if result.photo_url is not None:
        stop.photo_url = result.photo_url
    # Always write `photos` so the list stays in sync with the enrichment we
    # just resolved (a re-enrichment that returns fewer photos shouldn't leave
    # stale entries from the previous run). Cap at 3 — that's all the UI uses.
    primary = [result.photo_url] if result.photo_url else []
    stop.photos = (primary + result.additional_photos)[:3]
    if result.rating is not None:
        stop.rating = result.rating
    if result.formatted_address is not None:
        stop.formatted_address = result.formatted_address
    # Backfill coords only when missing — Claude's value wins if both have one.
    if stop.lat is None and result.lat is not None:
        stop.lat = result.lat
    if stop.lng is None and result.lng is not None:
        stop.lng = result.lng


async def _create_routes(
    db: AsyncSession,
    trip_id: UUID,
    days: Sequence[TripDay],
) -> tuple[int, int]:
    """Schedule one Directions call per consecutive (lat,lng)-bearing stop
    pair within the same day, plus one per consecutive-day transition
    (last geo'd stop of day N → first geo'd stop of day N+1). Cross-day
    routes are owned by the next day and rendered as dashed-grey on the
    client to read as "transit between days." Returns (attempted, created).
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

    # Cross-day pairs: last geo'd stop of day N → first geo'd stop of day N+1.
    # Owned by next_day so the route's trip_day_id matches the day the
    # traveler is heading INTO. Skipped when either side has no coords.
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

    if not pairs:
        return 0, 0

    tasks = []
    for _, prev, nxt in pairs:
        mode = (
            "walking"
            if haversine_km(prev.lat, prev.lng, nxt.lat, nxt.lng)
            < _WALKING_THRESHOLD_KM
            else "driving"
        )
        tasks.append(
            get_route(db, prev.lat, prev.lng, nxt.lat, nxt.lng, mode)
        )
    results = await asyncio.gather(*tasks, return_exceptions=True)

    created = 0
    for (day, prev, nxt), result in zip(pairs, results):
        if isinstance(result, Exception) or result is None:
            continue
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

    return build_static_map_url(
        api_key=settings.google_api_key,
        signing_secret=settings.google_url_signing_secret,
        markers_by_day=markers_by_day,
        polylines=polylines,
    )


async def geocode_stop_preview(
    *,
    name: str,
    city: str | None,
    lat: float | None,
    lng: float | None,
) -> tuple[float, float, str | None] | None:
    """Resolve one stop's coordinates for the live map, no persistence.

    Used by `/chat/stream` to drop a marker the instant a stop streams out
    of the itinerary tool call. Prefers coords Claude already supplied;
    otherwise runs the same Google Places text-search (`find_place_id_by_name`)
    every saved trip uses. Returns `(lat, lng, google_place_id|None)` or
    None when the stop can't be mapped. Never raises.
    """
    if lat is not None and lng is not None:
        return lat, lng, None
    if not (name or "").strip():
        return None
    try:
        result = await find_place_id_by_name(name=name, city=city)
    except Exception:
        logger.warning("geocode_stop_preview name=%r raised", name, exc_info=True)
        return None
    if result is None:
        return None
    place_id, r_lat, r_lng = result
    return r_lat, r_lng, place_id


async def geocode_itinerary_preview(db: AsyncSession, payload: dict) -> dict:
    """Geocode a draft itinerary WITHOUT persisting it, for the live map.

    Used by `/chat/stream` to light up the map the moment Claude emits an
    itinerary, before (and independent of) any "Save". Builds transient ORM
    objects from the validated `emit_itinerary` payload, runs the same
    `_enrich_places` resolution every saved trip goes through (Google Places
    text-search → google_place_id + lat/lng, plus photos/rating/address),
    and serializes the result into the canonical `TripOut` shape.

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
                    google_place_id=stop_payload.get("google_place_id"),
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

    # Local import dodges the repository ↔ enrichment cycle (repository
    # already imports this module lazily).
    from app.domain.trips.repository import _serialize

    return _serialize(trip)
