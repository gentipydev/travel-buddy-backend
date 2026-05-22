"""Unit tests for `app.domain.trips.scheduling`.

Pure functions, no DB. Covers the three building blocks that the mutations
layer composes when `recompute_times: true` arrives on a reorder or move
payload: travel-minute estimation, nearest-insertion index selection, and
forward-walk time stamping with the "skip when day has no anchor" rule.
"""
from __future__ import annotations

import uuid

from app.db.models import TripRoute, TripStop
from app.domain.trips.scheduling import (
    DEFAULT_DAY_START,
    DEFAULT_DURATIONS_BY_KIND,
    MIN_LEG_MINUTES,
    estimate_travel_minutes,
    forward_walk_times,
    nearest_insertion_index,
)


# Coordinates used across cases. Tirana / Berat / Vlorë anchor real distances
# so the haversine fallback produces interpretable minutes.
TIRANA = (41.3275, 19.8187)
BERAT = (40.7038, 19.9457)
VLORE = (40.4686, 19.4914)


def _make_stop(
    *,
    name: str = "Stop",
    kind: str = "sight",
    lat: float | None = None,
    lng: float | None = None,
    duration_minutes: int | None = None,
    arrival_time: str | None = None,
    end_time: str | None = None,
    stop_id: uuid.UUID | None = None,
) -> TripStop:
    """Build a TripStop with the fields the scheduler reads. FakeSession-free."""
    stop = TripStop(
        id=stop_id or uuid.uuid4(),
        sort_order=0,
        name=name,
        kind=kind,
    )
    stop.lat = lat
    stop.lng = lng
    stop.duration_minutes = duration_minutes
    stop.arrival_time = arrival_time
    stop.end_time = end_time
    return stop


def _make_route(*, duration_seconds: int | None) -> TripRoute:
    return TripRoute(
        id=uuid.uuid4(),
        encoded_polyline="x",
        distance_meters=None,
        duration_seconds=duration_seconds,
        travel_mode="driving",
    )


# ---------------------------------------------------------------------------
# estimate_travel_minutes
# ---------------------------------------------------------------------------


def test_estimate_uses_cached_route_when_available():
    route = _make_route(duration_seconds=1800)  # 30 min
    assert estimate_travel_minutes(*TIRANA, *BERAT, route) == 30


def test_estimate_floors_cached_route_below_min_leg():
    route = _make_route(duration_seconds=60)  # 1 min
    assert estimate_travel_minutes(*TIRANA, *TIRANA, route) == MIN_LEG_MINUTES


def test_estimate_ignores_cached_route_with_null_duration():
    """A TripRoute row with no duration_seconds (Directions API didn't
    return one) must NOT short-circuit — falls back to haversine."""
    route = _make_route(duration_seconds=None)
    result = estimate_travel_minutes(*TIRANA, *BERAT, route)
    assert result > MIN_LEG_MINUTES  # ~100km / 50kmh ≈ 120 min


def test_estimate_falls_back_to_walking_speed_under_threshold():
    """< WALKING_THRESHOLD_KM (2 km) → walking at 5 km/h."""
    # Two points ~1 km apart in Tirana centre.
    a = (41.3275, 19.8187)
    b = (41.3365, 19.8187)  # ~1.0 km north
    result = estimate_travel_minutes(*a, *b, None)
    # 1 km / 5 kmh = 12 min — well above MIN_LEG_MINUTES.
    assert 8 <= result <= 16


def test_estimate_falls_back_to_driving_speed_over_threshold():
    """≥ WALKING_THRESHOLD_KM → driving at DRIVING_AVG_KMH (50 km/h)."""
    # Tirana → Berat is ~70 km direct → ~84 min at 50 km/h.
    result = estimate_travel_minutes(*TIRANA, *BERAT, None)
    assert 60 <= result <= 110


def test_estimate_floor_when_coords_missing():
    """Either side missing coords → MIN_LEG_MINUTES (we don't guess)."""
    assert estimate_travel_minutes(None, None, *BERAT, None) == MIN_LEG_MINUTES
    assert estimate_travel_minutes(*TIRANA, None, None, None) == MIN_LEG_MINUTES
    assert estimate_travel_minutes(None, None, None, None, None) == MIN_LEG_MINUTES


# ---------------------------------------------------------------------------
# nearest_insertion_index
# ---------------------------------------------------------------------------


def test_nearest_insertion_into_empty_day_returns_0():
    new = _make_stop(lat=TIRANA[0], lng=TIRANA[1])
    assert nearest_insertion_index([], new, {}) == 0


def test_nearest_insertion_prefers_index_minimising_total_legs():
    """Day has Tirana → Vlorë. Inserting Berat in the middle (1) should
    win because Berat lies between them; appending at end (2) would
    require backtracking."""
    a = _make_stop(lat=TIRANA[0], lng=TIRANA[1])
    c = _make_stop(lat=VLORE[0], lng=VLORE[1])
    b = _make_stop(lat=BERAT[0], lng=BERAT[1])
    idx = nearest_insertion_index([a, c], b, {})
    assert idx == 1


def test_nearest_insertion_ties_broken_by_lower_index():
    """Both candidate positions have identical cost (single existing stop,
    new stop same coords) → return the earlier index."""
    a = _make_stop(lat=TIRANA[0], lng=TIRANA[1])
    new = _make_stop(lat=TIRANA[0], lng=TIRANA[1])
    # Two positions: 0 (before a) and 1 (after a). Both yield identical
    # total leg minutes (MIN_LEG_MINUTES from same-coord pair).
    idx = nearest_insertion_index([a], new, {})
    assert idx == 0


def test_nearest_insertion_uses_cached_routes_when_present():
    """When a cached row exists for a candidate adjacency, its duration
    overrides haversine. We make Tirana→Berat artificially cheap (5 min)
    so appending Berat at index 1 wins over inserting at 0."""
    a = _make_stop(lat=TIRANA[0], lng=TIRANA[1])
    b = _make_stop(lat=BERAT[0], lng=BERAT[1])
    cheap = _make_route(duration_seconds=300)  # 5 min
    routes = {(a.id, b.id): cheap}
    # Empty-existing list except for `a`, inserting `b`.
    # Index 0 → [b, a]: no cached route in routes for (b.id, a.id), falls
    #   back to haversine ~120min.
    # Index 1 → [a, b]: cached 5min via (a.id, b.id).
    idx = nearest_insertion_index([a], b, routes)
    assert idx == 1


# ---------------------------------------------------------------------------
# forward_walk_times
# ---------------------------------------------------------------------------


def test_forward_walk_returns_empty_when_no_stop_has_arrival_time():
    """The 'time-free trip' guard: if no stop on the day was timed before,
    the rescheduler must not impose times. This preserves explicit user
    intent (and Claude's emission when it deliberately omits times)."""
    a = _make_stop(arrival_time=None)
    b = _make_stop(arrival_time=None)
    assert forward_walk_times([a, b], {}) == []


def test_forward_walk_anchors_to_first_arrival_when_set():
    a = _make_stop(
        lat=TIRANA[0], lng=TIRANA[1],
        kind="sight", arrival_time="10:00", duration_minutes=60,
    )
    b = _make_stop(
        lat=TIRANA[0] + 0.005, lng=TIRANA[1],  # ~0.6 km — walking, MIN_LEG floor
        kind="meal", duration_minutes=60,
    )
    updates = forward_walk_times([a, b], {})
    assert len(updates) == 2
    # First stop keeps 10:00; end = 11:00 (60 min duration).
    assert updates[0][1]["arrival_time"] == "10:00"
    assert updates[0][1]["end_time"] == "11:00"
    assert updates[0][1]["duration_minutes"] == 60
    # Second stop arrives ~MIN_LEG_MINUTES (5) later → 11:05.
    assert updates[1][1]["arrival_time"] == "11:05"
    assert updates[1][1]["end_time"] == "12:05"


def test_forward_walk_falls_back_to_default_day_start_for_unset_first_arrival():
    """When the day has at least one stop with arrival_time but the FIRST
    stop's arrival_time is None, anchor falls to DEFAULT_DAY_START (09:00)."""
    a = _make_stop(
        lat=TIRANA[0], lng=TIRANA[1], kind="sight", arrival_time=None,
        duration_minutes=60,
    )
    b = _make_stop(
        lat=TIRANA[0], lng=TIRANA[1], kind="meal", arrival_time="12:00",
        duration_minutes=60,
    )
    updates = forward_walk_times([a, b], {})
    assert updates[0][1]["arrival_time"] == DEFAULT_DAY_START
    assert updates[0][1]["end_time"] == "10:00"


def test_forward_walk_preserves_explicit_duration_minutes():
    a = _make_stop(arrival_time="09:00", duration_minutes=45, kind="sight")
    updates = forward_walk_times([a], {})
    assert updates[0][1]["duration_minutes"] == 45
    assert updates[0][1]["end_time"] == "09:45"


def test_forward_walk_uses_kind_default_when_duration_missing():
    a = _make_stop(arrival_time="09:00", duration_minutes=None, kind="hike")
    updates = forward_walk_times([a], {})
    expected = DEFAULT_DURATIONS_BY_KIND["hike"]  # 180
    assert updates[0][1]["duration_minutes"] == expected
    assert updates[0][1]["end_time"] == "12:00"  # 09:00 + 180


def test_forward_walk_rounds_arrivals_to_5_minute_grid():
    """A stop with arrival_time '09:03' should round to '09:05' after walk."""
    a = _make_stop(arrival_time="09:03", duration_minutes=58, kind="sight")
    b = _make_stop(arrival_time=None, duration_minutes=30, kind="meal")
    updates = forward_walk_times([a, b], {})
    assert updates[0][1]["arrival_time"] == "09:05"
    # b arrives after a's end (10:03) + min-leg-floor 5min = 10:08 → round 10:10.
    assert updates[1][1]["arrival_time"] == "10:10"


def test_forward_walk_uses_cached_route_minutes_between_stops():
    a = _make_stop(arrival_time="09:00", duration_minutes=60, kind="sight")
    b = _make_stop(arrival_time=None, duration_minutes=60, kind="sight")
    routes = {(a.id, b.id): _make_route(duration_seconds=1800)}  # 30 min
    updates = forward_walk_times([a, b], routes)
    # a end = 10:00; + 30min travel = 10:30.
    assert updates[1][1]["arrival_time"] == "10:30"


def test_forward_walk_single_stop_just_stamps_self():
    a = _make_stop(arrival_time="14:00", duration_minutes=90, kind="sight")
    updates = forward_walk_times([a], {})
    assert len(updates) == 1
    assert updates[0][1]["arrival_time"] == "14:00"
    assert updates[0][1]["end_time"] == "15:30"


def test_forward_walk_clamps_overflow_past_2355():
    """A late-night stamp must clamp to 23:55 max so the persisted HH:MM
    stays inside the schema's validator range."""
    a = _make_stop(arrival_time="23:00", duration_minutes=120, kind="sight")
    b = _make_stop(arrival_time=None, duration_minutes=120, kind="sight")
    updates = forward_walk_times([a, b], {})
    # a arrival 23:00, end would be 25:00 → clamped 23:55.
    assert updates[0][1]["end_time"] == "23:55"
    # b arrival is well past midnight → also clamped.
    assert updates[1][1]["arrival_time"] == "23:55"
    assert updates[1][1]["end_time"] == "23:55"
