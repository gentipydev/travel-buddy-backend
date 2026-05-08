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
    cache an empty result — let the orchestrator draw a straight line and
    retry next save."""
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    mock_http.on(
        GOOGLE_URL,
        lambda req: httpx.Response(200, json={"status": "ZERO_RESULTS", "routes": []}),
    )

    row = await get_route(fake_db, 40.5, 19.5, 40.6, 19.6)

    assert row is None
    # Only the one read execution — no upsert path was triggered.
    assert len(fake_db.executed) == 1
