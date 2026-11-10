"""Deterministic re-scheduler for trip-day stops.

Pure functions, no DB I/O. Callers load cached `TripRoute` rows for a day
and pass them in; this module computes realistic times by forward-walking
through the day's ordering using either cached travel durations or a
haversine+speed fallback when no cached row exists for a pair.

Used by `app.domain.trips.mutations` when the client sets
`recompute_times: true` on a reorder or move payload. Refine tools (LLM-
driven) leave the flag at False so Claude's emitted times stay authoritative.
"""
from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import Any
from uuid import UUID

from app.db.models import TripRoute, TripStop
from app.domain.trips.geo import haversine_km

logger = logging.getLogger(__name__)


# Per-kind defaults used only when a stop has no explicit duration_minutes.
# Matches the StopKind literal in schemas.py / mobile entities.
DEFAULT_DURATIONS_BY_KIND: dict[str, int] = {
    "sight": 90,
    "beach": 120,
    "hike": 180,
    "meal": 60,
    "stay": 30,
    "transport": 15,
    "activity": 90,
    "unknown": 60,
}

DEFAULT_DAY_START = "09:00"

# Conservative Albanian rural driving average. Used as the fallback when no
# cached route exists for a coord pair and haversine ≥ WALKING_THRESHOLD_KM.
DRIVING_AVG_KMH = 50.0

# Match the constant in app/domain/trips/enrichment.py so this module agrees
# with how enrichment classifies legs as walking vs driving.
WALKING_THRESHOLD_KM = 2.0
WALKING_AVG_KMH = 5.0

# Floor on per-leg travel so two consecutive stops at the same coords don't
# both anchor at the same arrival_time.
MIN_LEG_MINUTES = 5

# Round arrivals to a 5-min grid so users see clean times.
TIME_ROUNDING_MIN = 5

# Hard cap so the HH:MM string we persist stays inside the validator's
# allowed range (max 23:59). Overflow past 23:55 is logged so we can
# monitor how often days don't fit in a single calendar day.
MAX_MINUTES_OF_DAY = 23 * 60 + 55


def _parse_hhmm(value: str | None) -> int | None:
    """'HH:MM' → minutes-since-midnight. None on null/invalid."""
    if not value:
        return None
    try:
        hour_str, minute_str = value.split(":", 1)
        hour = int(hour_str)
        minute = int(minute_str)
    except (ValueError, AttributeError):
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return None
    return hour * 60 + minute


def _format_hhmm(minutes: int) -> str:
    """Minutes-since-midnight → 'HH:MM', clamped to [00:00, 23:55]."""
    clamped = max(0, min(minutes, MAX_MINUTES_OF_DAY))
    return f"{clamped // 60:02d}:{clamped % 60:02d}"


def _round_to_grid(minutes: int) -> int:
    """Round to the nearest TIME_ROUNDING_MIN minutes (banker's-round-down)."""
    return ((minutes + TIME_ROUNDING_MIN // 2) // TIME_ROUNDING_MIN) * TIME_ROUNDING_MIN


def _duration_for_stop(stop: TripStop | Any) -> int:
    """Preserve explicit duration_minutes; fall back to kind-default; else 60."""
    explicit = getattr(stop, "duration_minutes", None)
    if isinstance(explicit, int) and explicit > 0:
        return explicit
    kind = getattr(stop, "kind", None) or "unknown"
    return DEFAULT_DURATIONS_BY_KIND.get(kind, DEFAULT_DURATIONS_BY_KIND["unknown"])


def estimate_travel_minutes(
    from_lat: float | None,
    from_lng: float | None,
    to_lat: float | None,
    to_lng: float | None,
    cached_route: TripRoute | None,
) -> int:
    """Best estimate of travel time between two coords, in minutes.

    Preference order:
      1. cached_route.duration_seconds (Mapbox Directions, road-snapped)
      2. haversine km + speed-by-mode (walking < WALKING_THRESHOLD_KM else driving)
      3. MIN_LEG_MINUTES floor when coords are missing entirely

    Always returns at least MIN_LEG_MINUTES so consecutive same-coord stops
    don't share an arrival_time.
    """
    if cached_route is not None and cached_route.duration_seconds:
        minutes = max(1, round(cached_route.duration_seconds / 60))
        return max(MIN_LEG_MINUTES, minutes)

    if from_lat is None or from_lng is None or to_lat is None or to_lng is None:
        return MIN_LEG_MINUTES

    km = haversine_km(from_lat, from_lng, to_lat, to_lng)
    speed = WALKING_AVG_KMH if km < WALKING_THRESHOLD_KM else DRIVING_AVG_KMH
    minutes = round((km / speed) * 60)
    return max(MIN_LEG_MINUTES, minutes)


def _route_lookup(
    routes_by_pair: Mapping[tuple[UUID, UUID], TripRoute],
    from_stop: TripStop | Any,
    to_stop: TripStop | Any,
) -> TripRoute | None:
    return routes_by_pair.get((from_stop.id, to_stop.id))


def _leg_minutes(
    from_stop: TripStop | Any,
    to_stop: TripStop | Any,
    routes_by_pair: Mapping[tuple[UUID, UUID], TripRoute],
) -> int:
    return estimate_travel_minutes(
        getattr(from_stop, "lat", None),
        getattr(from_stop, "lng", None),
        getattr(to_stop, "lat", None),
        getattr(to_stop, "lng", None),
        _route_lookup(routes_by_pair, from_stop, to_stop),
    )


def nearest_insertion_index(
    day_stops: Sequence[TripStop | Any],
    new_stop: TripStop | Any,
    routes_by_pair: Mapping[tuple[UUID, UUID], TripRoute],
) -> int:
    """Pick the insertion index that minimises total travel duration.

    Tries positions 0..len(day_stops). The cost of inserting at position k is
    (leg into k) + (leg out of k) − (replaced leg), so we just sum all legs
    of the resulting order and pick the minimum. Ties broken by lower index.

    Uses `routes_by_pair` (cached `TripRoute` rows) when available and falls
    back to haversine+speed; never fires a network call. Safe for the
    drag-drop hot path.
    """
    if not day_stops:
        return 0

    def total_legs(stops_seq: Sequence[Any]) -> int:
        total = 0
        for a, b in zip(stops_seq, stops_seq[1:]):
            total += _leg_minutes(a, b, routes_by_pair)
        return total

    best_index = 0
    best_cost: int | None = None
    for k in range(len(day_stops) + 1):
        candidate = [*day_stops[:k], new_stop, *day_stops[k:]]
        cost = total_legs(candidate)
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_index = k
    return best_index


def forward_walk_times(
    ordered_stops: Sequence[TripStop | Any],
    routes_by_pair: Mapping[tuple[UUID, UUID], TripRoute],
) -> list[tuple[UUID, dict]]:
    """Compute realistic times for a day's stops in the given order.

    Returns [(stop_id, {arrival_time, end_time, duration_minutes}), ...]
    one entry per input stop. Returns [] when NO stop in the day has an
    `arrival_time` set — preserves the "time-free trip" pattern so the
    rescheduler never imposes times on trips the user (or AI) deliberately
    left unscheduled.

    Anchor: first stop's existing arrival_time when present, else DEFAULT_DAY_START.
    Per-stop duration: preserved when explicit; otherwise DEFAULT_DURATIONS_BY_KIND.
    Travel between stops: via `estimate_travel_minutes` (cached routes preferred).
    Arrivals rounded to TIME_ROUNDING_MIN grid; clamped to 23:55.
    """
    if not ordered_stops:
        return []

    any_arrival = any(getattr(s, "arrival_time", None) for s in ordered_stops)
    if not any_arrival:
        return []

    first_arrival = _parse_hhmm(getattr(ordered_stops[0], "arrival_time", None))
    if first_arrival is None:
        first_arrival = _parse_hhmm(DEFAULT_DAY_START)
    assert first_arrival is not None  # DEFAULT_DAY_START is always valid

    updates: list[tuple[UUID, dict]] = []
    current_minutes = _round_to_grid(first_arrival)
    overflowed = False

    for index, stop in enumerate(ordered_stops):
        if index > 0:
            prev = ordered_stops[index - 1]
            current_minutes += _leg_minutes(prev, stop, routes_by_pair)
            current_minutes = _round_to_grid(current_minutes)

        if current_minutes > MAX_MINUTES_OF_DAY and not overflowed:
            overflowed = True
            logger.warning(
                "scheduling.forward_walk_times overflow past 23:55 at stop %s",
                getattr(stop, "id", None),
            )

        arrival = _format_hhmm(current_minutes)
        duration = _duration_for_stop(stop)
        end_minutes = current_minutes + duration
        end_time = _format_hhmm(end_minutes)

        updates.append(
            (
                stop.id,
                {
                    "arrival_time": arrival,
                    "end_time": end_time,
                    "duration_minutes": duration,
                },
            )
        )
        current_minutes = end_minutes

    return updates
