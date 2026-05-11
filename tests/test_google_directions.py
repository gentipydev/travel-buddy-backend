"""Unit tests for app.services.google_directions.get_route.

Same shape as the places tests minus the curated tier. Cache → API, kill
switches, failure modes.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import httpx

from app.db.models import DirectionsCache
from app.services import google_directions
from app.services.google_directions import _cache_key, get_route


GOOGLE_URL = "https://maps.googleapis.com/maps/api/directions/json"
ROADS_URL = "https://roads.googleapis.com/v1/nearestRoads"


def _stub_settings(monkeypatch, *, key="dummy-key", directions_enabled=True):
    monkeypatch.setattr(google_directions.settings, "google_api_key", key)
    monkeypatch.setattr(
        google_directions.settings, "google_directions_enabled", directions_enabled
    )


def _cached_row(key: str) -> DirectionsCache:
    return DirectionsCache(
        cache_key=key,
        encoded_polyline="cached_polyline",
        distance_meters=1200,
        duration_seconds=180,
        travel_mode="driving",
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )


def _directions_response(polyline="abc123") -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "OK",
            "routes": [
                {
                    "overview_polyline": {"points": polyline},
                    "legs": [
                        {
                            "distance": {"value": 4500, "text": "4.5 km"},
                            "duration": {"value": 540, "text": "9 mins"},
                        }
                    ],
                }
            ],
        },
    )


def test_cache_key_rounds_to_4_decimals():
    # Two trips that differ only at the 6th decimal share a key (the 4th
    # decimal is preserved; differences smaller than ~10m collapse).
    k1 = _cache_key(40.703810, 19.945720, 41.327500, 19.818700, "driving")
    k2 = _cache_key(40.703815, 19.945723, 41.327503, 19.818701, "driving")
    assert k1 == k2
    # A 4th-decimal difference *does* matter — different keys.
    k3 = _cache_key(40.7039, 19.9457, 41.3275, 19.8187, "driving")
    assert k1 != k3


async def test_cache_hit_skips_api(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    key = _cache_key(40.5, 19.5, 40.6, 19.6, "driving")
    fake_db.queue(_cached_row(key))

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6, "driving")

    assert row is not None
    assert row.encoded_polyline == "cached_polyline"
    assert mock_http.requests == []


async def test_fresh_fetch_writes_cache(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    key = _cache_key(40.5, 19.5, 40.6, 19.6, "driving")
    # Three execute() calls in this path: SELECT for cache-miss, the
    # INSERT … ON CONFLICT (return value ignored), and the post-upsert
    # SELECT that returns the managed instance.
    fake_db.queue(None)             # cache miss
    fake_db.queue(None)             # upsert returns nothing the code reads
    fake_db.queue(_cached_row(key)) # re-read after upsert
    mock_http.on(GOOGLE_URL, lambda req: _directions_response("fresh_poly"))

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6, "driving")

    assert row is not None
    assert len(mock_http.requests) == 1
    assert len(fake_db.executed) == 3


async def test_disabled_returns_none(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch, directions_enabled=False)

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6, "driving")

    assert row is None
    assert mock_http.requests == []
    # Kill switch short-circuits before any DB query.
    assert fake_db.executed == []


async def test_empty_api_key_treated_as_disabled(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch, key="")

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None
    assert mock_http.requests == []


async def test_api_error_returns_none(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    mock_http.on(GOOGLE_URL, lambda req: httpx.Response(500))

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None


async def test_quota_exceeded_logs_clearly(
    fake_db, mock_http, monkeypatch, caplog
):
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    mock_http.on(GOOGLE_URL, lambda req: httpx.Response(429))

    with caplog.at_level(logging.WARNING, logger="app.services.google_directions"):
        row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None
    assert any("quota" in rec.getMessage() for rec in caplog.records)


async def test_zero_routes_returns_none_without_caching(
    fake_db, mock_http, monkeypatch
):
    """Directions API can return ZERO_RESULTS (e.g. unreachable). Don't
    cache an empty result. Snap-retry is then attempted; if Roads returns
    nothing too, the final answer is None."""
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    mock_http.on(
        GOOGLE_URL,
        lambda req: httpx.Response(200, json={"status": "ZERO_RESULTS", "routes": []}),
    )
    # Roads API also yields nothing → no retry possible, final None.
    mock_http.on(
        ROADS_URL,
        lambda req: httpx.Response(200, json={"snappedPoints": []}),
    )

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None
    # Only the one read execution — neither original nor snapped fetch upserted.
    assert len(fake_db.executed) == 1


def _roads_response(snapped: list[tuple[int, float, float]]) -> httpx.Response:
    """Build a Roads API response. [snapped] = list of (originalIndex, lat, lng)."""
    return httpx.Response(
        200,
        json={
            "snappedPoints": [
                {
                    "location": {"latitude": lat, "longitude": lng},
                    "originalIndex": idx,
                    "placeId": f"road_{idx}",
                }
                for idx, lat, lng in snapped
            ]
        },
    )


async def test_snap_retry_succeeds_when_first_directions_fails(
    fake_db, mock_http, monkeypatch
):
    """First Directions call ZERO_RESULTS → snap both endpoints → retry
    Directions with snapped coords → success cached under the ORIGINAL key."""
    _stub_settings(monkeypatch)
    key = _cache_key(40.5, 19.5, 40.6, 19.6, "driving")
    # Cache miss (1), upsert (2), re-read after upsert (3) — same three-execute
    # sequence as test_fresh_fetch_writes_cache.
    fake_db.queue(None)
    fake_db.queue(None)
    fake_db.queue(_cached_row(key))

    directions_calls = []

    def directions_handler(req):
        directions_calls.append(req)
        # First call returns ZERO_RESULTS, second (after snap) returns success.
        if len(directions_calls) == 1:
            return httpx.Response(
                200, json={"status": "ZERO_RESULTS", "routes": []}
            )
        return _directions_response("snapped_poly")

    mock_http.on(GOOGLE_URL, directions_handler)
    # Snap moves each endpoint ~10m (well under 250m).
    mock_http.on(
        ROADS_URL,
        lambda req: _roads_response(
            [(0, 40.50009, 19.50009), (1, 40.60009, 19.60009)]
        ),
    )

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6, "driving")

    assert row is not None
    # Two Directions calls (original + retry) + one Roads call.
    assert len(directions_calls) == 2
    assert len(mock_http.requests) == 3
    # Three DB executes confirms upsert happened (under the original key).
    assert len(fake_db.executed) == 3


async def test_snap_retry_skipped_when_first_directions_succeeds(
    fake_db, mock_http, monkeypatch
):
    """Sanity: no Roads API call when the original Directions call works."""
    _stub_settings(monkeypatch)
    key = _cache_key(40.5, 19.5, 40.6, 19.6, "driving")
    fake_db.queue(None)
    fake_db.queue(None)
    fake_db.queue(_cached_row(key))
    mock_http.on(GOOGLE_URL, lambda req: _directions_response("first_try"))
    # Register a Roads handler that would fail the test if invoked.
    roads_calls: list = []

    def roads_handler(req):
        roads_calls.append(req)
        return httpx.Response(500)

    mock_http.on(ROADS_URL, roads_handler)

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is not None
    assert roads_calls == []


async def test_snap_too_far_returns_none(fake_db, mock_http, monkeypatch):
    """If the Roads API moves an endpoint > _MAX_SNAP_METERS from its
    original coord, reject the retry — a polyline starting visibly off-
    marker is worse than no polyline."""
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    directions_calls: list = []

    def directions_handler(req):
        directions_calls.append(req)
        return httpx.Response(200, json={"status": "ZERO_RESULTS", "routes": []})

    mock_http.on(GOOGLE_URL, directions_handler)
    # Move the FROM endpoint ~7 km from the original — well past 250m.
    mock_http.on(
        ROADS_URL,
        lambda req: _roads_response(
            [(0, 40.56, 19.56), (1, 40.60009, 19.60009)]
        ),
    )

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None
    # Only the original Directions call — the snapped retry was rejected.
    assert len(directions_calls) == 1
    # No upsert path was triggered, only the initial cache-miss read.
    assert len(fake_db.executed) == 1


async def test_snap_api_failure_returns_none(fake_db, mock_http, monkeypatch):
    """If the Roads API itself errors (e.g. 403 because Roads isn't
    enabled in GCP), the retry path bails cleanly with None."""
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    mock_http.on(
        GOOGLE_URL,
        lambda req: httpx.Response(200, json={"status": "ZERO_RESULTS", "routes": []}),
    )
    mock_http.on(ROADS_URL, lambda req: httpx.Response(403))

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None
    assert len(fake_db.executed) == 1
