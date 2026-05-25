"""Trip mutation primitives shared by the trips router and the chat refine
tools.

Each `apply_*` function loads + mutates + flushes the trip but does NOT
commit — the caller owns the transaction boundary. This lets a refine
turn apply tool ops and persist the assistant message in one commit, and
lets the trips router's HTTP handlers wrap whole requests in a single
transaction.

`replace_trip_from_payload` is the full re-emit path (Claude's
`emit_itinerary` tool); the granular `apply_*` functions are the surgical
patch tools.
"""
import logging
from datetime import date
from uuid import UUID

from fastapi import HTTPException
from sqlalchemy import case, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Trip, TripDay, TripRoute, TripStop, utcnow
from app.domain.trips.repository import (
    _load_day,
    _load_stop,
    _load_trip,
    _touch_trip,
)
from app.domain.trips.scheduling import (
    forward_walk_times,
    nearest_insertion_index,
)


logger = logging.getLogger(__name__)


async def _load_route_cache_for_day(
    db: AsyncSession, day_id: UUID,
) -> dict[tuple[UUID, UUID], TripRoute]:
    """Load `trip_routes` rows for a day, keyed by (from_stop_id, to_stop_id).

    Cache hits cover pairs that were consecutive on the day before the
    current mutation; newly-adjacent pairs miss and fall back to haversine
    inside the scheduler. `enrich_trip` will repopulate the cache for the
    post-mutation adjacencies during the serialize step that runs after
    this mutation commits, so subsequent reorders pay no repeat Directions
    cost on the same pairs.
    """
    stmt = select(TripRoute).where(TripRoute.trip_day_id == day_id)
    rows = (await db.execute(stmt)).scalars().all()
    return {(r.from_stop_id, r.to_stop_id): r for r in rows}


def _apply_forward_walk_updates(
    ordered_stops: list[TripStop],
    routes_by_pair: dict[tuple[UUID, UUID], TripRoute],
) -> None:
    """Compute realistic times for `ordered_stops` and mutate each stop's
    `arrival_time`, `end_time`, `duration_minutes` via ORM setattr (same
    pattern as `apply_update_stop`). SQLAlchemy will flush these as one
    UPDATE per dirtied row at the next `db.flush()`. No-op when the day
    has no anchored times (forward_walk_times returns [] for time-free
    days)."""
    updates = forward_walk_times(ordered_stops, routes_by_pair)
    by_id = {s.id: s for s in ordered_stops}
    for stop_id, fields in updates:
        stop_obj = by_id.get(stop_id)
        if stop_obj is None:
            continue
        for key, value in fields.items():
            setattr(stop_obj, key, value)


async def replace_trip_from_payload(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    payload: dict,
) -> Trip:
    """Atomically replace a trip's days/stops from a validated TripCreate dict.

    Caller owns the transaction — this function flushes but does not commit, so
    a refine turn can land the trip replace and the assistant-message persist
    in one boundary. Raises HTTPException(404) if the trip is not owned by
    `user_id`.
    """
    trip = await _load_trip(db, trip_id, user_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")

    trip.title = payload["title"]
    trip.summary = payload.get("summary")
    sd = payload.get("start_date")
    trip.start_date = date.fromisoformat(sd) if sd else None
    trip.updated_at = utcnow()

    # Clear existing children — cascade="all, delete-orphan" wipes days then
    # stops as the assignment runs. Flush before re-adding so the
    # UNIQUE(trip_id, day_number) and UNIQUE(trip_day_id, sort_order)
    # constraints don't see overlapping old + new rows mid-transaction.
    trip.days = []
    await db.flush()

    for day in payload["days"]:
        new_day = TripDay(
            day_number=day["day_number"],
            title=day.get("title"),
            base_city=day.get("base_city"),
        )
        for idx, stop in enumerate(day["stops"]):
            new_day.stops.append(
                TripStop(
                    sort_order=idx,
                    name=stop["name"],
                    kind=stop["kind"],
                    city=stop.get("city"),
                    address=stop.get("address"),
                    duration_minutes=stop.get("duration_minutes"),
                    arrival_time=stop.get("arrival_time"),
                    end_time=stop.get("end_time"),
                    transportation=stop.get("transportation"),
                    notes=stop.get("notes"),
                    lat=stop.get("lat"),
                    lng=stop.get("lng"),
                    google_place_id=stop.get("google_place_id"),
                )
            )
        trip.days.append(new_day)
    await db.flush()

    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        # Should be impossible — we just flushed it.
        raise HTTPException(status_code=500, detail="Trip vanished after replace")
    return refreshed


async def apply_update_trip(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    payload: dict,
) -> Trip:
    """Patch trip-level fields (title / summary / start_date). `payload` is the
    Pydantic `TripUpdate.model_dump(exclude_unset=True)` — only keys present
    are written. No-commit: caller owns the transaction."""
    trip = await _load_trip(db, trip_id, user_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    for key, value in payload.items():
        setattr(trip, key, value)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update")
    return refreshed


async def apply_add_day(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    payload: dict,
) -> Trip:
    """Insert a new day. `payload` matches `TripDayCreateBody.model_dump()`:
    `day_number` (required), optional `title` / `base_city`, and a `stops` list
    (may be empty). Raises HTTPException(409) on duplicate `day_number`."""
    trip = await _load_trip(db, trip_id, user_id)
    if trip is None:
        raise HTTPException(status_code=404, detail="Trip not found")
    if any(d.day_number == payload["day_number"] for d in trip.days):
        raise HTTPException(
            status_code=409, detail="day_number already exists"
        )
    new_day = TripDay(
        trip_id=trip.id,
        day_number=payload["day_number"],
        title=payload.get("title"),
        base_city=payload.get("base_city"),
    )
    for index, stop in enumerate(payload.get("stops") or []):
        new_day.stops.append(
            TripStop(
                sort_order=index,
                name=stop["name"],
                kind=stop["kind"],
                city=stop.get("city"),
                address=stop.get("address"),
                duration_minutes=stop.get("duration_minutes"),
                arrival_time=stop.get("arrival_time"),
                end_time=stop.get("end_time"),
                transportation=stop.get("transportation"),
                notes=stop.get("notes"),
                lat=stop.get("lat"),
                lng=stop.get("lng"),
                google_place_id=stop.get("google_place_id"),
            )
        )
    db.add(new_day)
    await _touch_trip(db, trip.id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after add_day")
    return refreshed


async def apply_update_day(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
    payload: dict,
) -> Trip:
    """Patch a day's metadata (title / base_city). `payload` is
    `TripDayUpdate.model_dump(exclude_unset=True)` — only set keys are
    written. day_number is NOT patchable here (renumbering days in place
    would silently break UI labels and stop time anchors). Use add+delete
    if you really need a different number."""
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    for key, value in payload.items():
        setattr(day, key, value)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update_day")
    return refreshed


async def apply_delete_day(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
) -> Trip:
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    await db.delete(day)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after delete_day")
    return refreshed


async def apply_insert_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
    payload: dict,
) -> Trip:
    """Insert a stop. `payload` matches `TripStopCreateBody.model_dump()`:
    all `TripStopCreate` fields plus optional `sort_order`. Omit `sort_order`
    to append; provide it to insert at a position (existing stops at >= that
    index are shifted up by 1 in a single UPDATE)."""
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    body = dict(payload)
    requested = body.pop("sort_order", None)
    if requested is None:
        next_sort = max((s.sort_order for s in day.stops), default=-1) + 1
        new_stop = TripStop(trip_day_id=day.id, sort_order=next_sort, **body)
    else:
        await db.execute(
            update(TripStop)
            .where(
                TripStop.trip_day_id == day.id,
                TripStop.sort_order >= requested,
            )
            .values(sort_order=TripStop.sort_order + 1)
        )
        new_stop = TripStop(trip_day_id=day.id, sort_order=requested, **body)
    db.add(new_stop)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after insert_stop")
    return refreshed


async def apply_update_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    stop_id: UUID,
    payload: dict,
) -> Trip:
    """Patch a single stop's fields. `payload` is
    `TripStopUpdate.model_dump(exclude_unset=True)` — only set keys are
    written, leaving everything else (including google_place_id, lat, lng,
    sort_order) untouched."""
    stop = await _load_stop(db, trip_id, stop_id, user_id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    for key, value in payload.items():
        setattr(stop, key, value)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after update_stop")
    return refreshed


async def apply_delete_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    stop_id: UUID,
) -> Trip:
    stop = await _load_stop(db, trip_id, stop_id, user_id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    # Per design: do NOT compact remaining sort_orders on delete. Gaps are
    # fine; the relationship's `order_by="TripStop.sort_order"` handles it.
    await db.delete(stop)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after delete_stop")
    return refreshed


async def apply_reorder_stops(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    day_id: UUID,
    payload: dict,
) -> Trip:
    """Reorder stops within a day. `payload` matches
    `StopReorderBody.model_dump(exclude_unset=True)` with
    `order: list[{stop_id, sort_order, arrival_time?, end_time?, duration_minutes?}]`.
    Caller must send the COMPLETE new ordering so the server can validate the
    set matches what's on disk. Optional per-item time fields are only written
    when the caller explicitly included them (legacy {stop_id, sort_order}-only
    callers leave times untouched)."""
    day = await _load_day(db, trip_id, day_id, user_id)
    if day is None:
        raise HTTPException(status_code=404, detail="Day not found")
    order_items = payload.get("order") or []
    # Coerce stop_id strings → UUID for set comparison parity with ORM ids.
    # Each tuple carries the per-item time-field dict (only keys actually
    # present in the payload; absent keys mean "leave column untouched", a
    # null value means "clear the column").
    parsed_order: list[tuple[UUID, int, dict]] = []
    for item in order_items:
        sid = UUID(str(item["stop_id"]))
        so = int(item["sort_order"])
        time_fields = {
            k: item[k]
            for k in ("arrival_time", "end_time", "duration_minutes")
            if k in item
        }
        parsed_order.append((sid, so, time_fields))
    current_ids = {s.id for s in day.stops}
    body_ids = {sid for sid, _, _ in parsed_order}
    if current_ids != body_ids:
        raise HTTPException(status_code=409, detail="stop set out of sync")
    body_orders = sorted(so for _, so, _ in parsed_order)
    if body_orders != list(range(len(body_orders))):
        raise HTTPException(
            status_code=400,
            detail="sort_order values must be 0..N-1 contiguous",
        )
    new_order = {sid: so for sid, so, _ in parsed_order}
    # Two-phase update to dodge mid-statement UNIQUE(trip_day_id, sort_order)
    # violations on a permutation. Despite the docs implying constraints are
    # checked at end-of-statement, PostgreSQL evaluates non-deferrable UNIQUE
    # per-row during UPDATE — so a swap (e.g. row at sort_order=1 → 2 while
    # row at sort_order=2 still sits at 2) fails the moment the first row
    # collides. Shifting every stop in the day into a disjoint negative range
    # first guarantees the positive 0..N-1 target range is free for phase 2.
    # Validation above ensures `current_ids == body_ids`, so every row in the
    # day is addressed by the CASE in phase 2 (no row gets NULL from a
    # missing WHEN branch).
    SHIFT = 1_000_000
    await db.execute(
        update(TripStop)
        .where(TripStop.trip_day_id == day.id)
        .values(sort_order=TripStop.sort_order - SHIFT)
    )
    await db.execute(
        update(TripStop)
        .where(TripStop.trip_day_id == day.id)
        .values(sort_order=case(new_order, value=TripStop.id))
    )
    if payload.get("recompute_times"):
        # Server-authoritative reschedule: per-item time fields in `order`
        # are ignored, and times are forward-walked across the new ordering
        # using cached travel durations (haversine + speed when no cache).
        ordered_stops_after = sorted(day.stops, key=lambda s: new_order[s.id])
        routes_by_pair = await _load_route_cache_for_day(db, day.id)
        _apply_forward_walk_updates(ordered_stops_after, routes_by_pair)
    else:
        # Legacy slot-anchored path: per-item time fields land verbatim.
        for sid, _, time_fields in parsed_order:
            if not time_fields:
                continue
            await db.execute(
                update(TripStop).where(TripStop.id == sid).values(**time_fields)
            )
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after reorder")
    return refreshed


async def apply_move_stop(
    db: AsyncSession,
    *,
    trip_id: UUID,
    user_id: int,
    stop_id: UUID,
    payload: dict,
) -> Trip:
    """Move a stop to a different day within the same trip. `payload` matches
    `StopMoveBody.model_dump()`: `target_day_id` (required), optional
    `sort_order` for target position (omit or pass beyond target length to
    append).

    Unlike `apply_delete_stop`, the source day's `sort_order` values ARE
    compacted — move is a rearrange, not a deletion. The target day shifts
    existing stops at >= `sort_order` up by one to make room. All stop
    fields (id, arrival_time, end_time, duration_minutes, completed_at,
    enrichment, etc.) survive the move; only `trip_day_id` and `sort_order`
    change. Same-day moves are rejected with 400 — use reorder instead.
    """
    stop = await _load_stop(db, trip_id, stop_id, user_id)
    if stop is None:
        raise HTTPException(status_code=404, detail="Stop not found")
    try:
        target_day_id = UUID(str(payload["target_day_id"]))
    except (KeyError, ValueError) as exc:
        raise HTTPException(
            status_code=400, detail="target_day_id is required"
        ) from exc
    target_day = await _load_day(db, trip_id, target_day_id, user_id)
    if target_day is None:
        raise HTTPException(status_code=404, detail="Target day not found")
    if stop.trip_day_id == target_day.id:
        raise HTTPException(
            status_code=400,
            detail="Stop is already on the target day; use reorder",
        )

    source_day_id = stop.trip_day_id
    old_sort = stop.sort_order
    target_len = len(target_day.stops)
    requested = payload.get("sort_order")
    auto_position = bool(payload.get("auto_position"))
    recompute = bool(payload.get("recompute_times"))

    # Load cached routes once when either operation needs them; both
    # `nearest_insertion_index` (pre-move) and the destination reschedule
    # (post-move) consult the same (from_stop_id, to_stop_id) cache.
    routes_by_pair: dict[tuple[UUID, UUID], TripRoute] = {}
    if auto_position or recompute:
        routes_by_pair = await _load_route_cache_for_day(db, target_day.id)

    if requested is None:
        if auto_position and target_day.stops:
            # Pick the insertion index that minimises total leg minutes on
            # the destination day. Source day is intentionally untouched.
            new_sort = nearest_insertion_index(
                target_day.stops, stop, routes_by_pair
            )
        else:
            new_sort = target_len
    else:
        requested_int = int(requested)
        if requested_int < 0:
            raise HTTPException(
                status_code=400, detail="sort_order must be >= 0"
            )
        new_sort = min(requested_int, target_len)

    # Three-phase update to dodge UNIQUE(trip_day_id, sort_order) collisions
    # while the moved stop, the source compaction, and the target shift are
    # all in flight. Same trick as `apply_reorder_stops` — shift the moved
    # stop into a disjoint negative range first, then rearrange the two days
    # in their now-collision-free positive ranges.
    #
    # synchronize_session=False on all four statements: these are bulk SQL
    # UPDATEs and the default "evaluate" strategy would mutate in-memory ORM
    # objects to match, corrupting the sort_order bookkeeping we do below.
    SHIFT = 1_000_000
    await db.execute(
        update(TripStop)
        .where(TripStop.id == stop.id)
        .values(sort_order=TripStop.sort_order - SHIFT)
        .execution_options(synchronize_session=False)
    )
    await db.execute(
        update(TripStop)
        .where(
            TripStop.trip_day_id == source_day_id,
            TripStop.sort_order > old_sort,
        )
        .values(sort_order=TripStop.sort_order - 1)
        .execution_options(synchronize_session=False)
    )
    # Phase 3: free `new_sort` on target_day by shifting stops at >= new_sort
    # up by 1. Can't be a single bulk `sort_order + 1` UPDATE — Postgres
    # checks UNIQUE constraints per row (not at statement end), and the
    # unique index scans rows ascending, so `1 → 2` collides with the row
    # still sitting at 2. Iterating per-row in DESCENDING order guarantees
    # each shift lands in a slot the previous shift just emptied. Also
    # naturally handles sort_order gaps from prior deletions: iterating
    # matched stops covers `new_sort == target_len` collisions too.
    for shifted_stop in sorted(
        [s for s in target_day.stops if s.sort_order >= new_sort],
        key=lambda s: s.sort_order,
        reverse=True,
    ):
        await db.execute(
            update(TripStop)
            .where(TripStop.id == shifted_stop.id)
            .values(sort_order=shifted_stop.sort_order + 1)
            .execution_options(synchronize_session=False)
        )
    await db.execute(
        update(TripStop)
        .where(TripStop.id == stop.id)
        .values(trip_day_id=target_day.id, sort_order=new_sort)
        .execution_options(synchronize_session=False)
    )
    # Sync the moved stop's in-memory fields so any later mutation that
    # reads `stop.sort_order` / `stop.trip_day_id` sees the post-move state.
    # Because synchronize_session=False was used above, Python objects still
    # hold their pre-move sort_orders — partition on those to build target_after.
    stop.sort_order = new_sort
    stop.trip_day_id = target_day.id
    if recompute:
        target_after = (
            [s for s in target_day.stops if s.sort_order < new_sort]
            + [stop]
            + [s for s in target_day.stops if s.sort_order >= new_sort]
        )
        _apply_forward_walk_updates(target_after, routes_by_pair)
    await _touch_trip(db, trip_id)
    await db.flush()
    refreshed = await _load_trip(db, trip_id, user_id)
    if refreshed is None:
        raise HTTPException(status_code=500, detail="Trip vanished after move_stop")
    return refreshed
