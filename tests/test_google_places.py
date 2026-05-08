"""Unit tests for app.services.google_places.enrich_stop.

The three tier paths (curated → cache → fresh), plus the kill-switch and
failure-mode contracts. Mocks the HTTP layer via the conftest registry; the
DB layer is a `FakeSession` that returns scripted SELECT results.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from uuid import uuid4

import httpx
import pytest

from app.db.models import Place, PlaceCache
from app.services import google_places
from app.services.google_places import enrich_stop


GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"


def _stub_settings(monkeypatch, *, key="dummy-key", places_enabled=True):
    monkeypatch.setattr(google_places.settings, "google_api_key", key)
    monkeypatch.setattr(
        google_places.settings, "google_places_enabled", places_enabled
    )


def _curated_place(google_place_id="ChIJtest") -> Place:
    return Place(
        id=uuid4(),
        google_place_id=google_place_id,
        name="Berat Castle",
        city="Berat",
        photos=["https://supabase.example/photos/berat-castle.jpg"],
        rating=4.7,
        latitude=40.7038,
        longitude=19.9457,
    )


def _cached_place(google_place_id="ChIJtest") -> PlaceCache:
    return PlaceCache(
        google_place_id=google_place_id,
        name="Berat Castle",
        formatted_address="Berat Castle, Berat, Albania",
        lat=40.7038,
        lng=19.9457,
        rating=4.5,
        primary_photo_url="https://lh3.googleusercontent.example/cached.jpg",
        expires_at=datetime.now(timezone.utc) + timedelta(days=30),
    )


def _details_response(photo_refs: list[str] | None = None) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "OK",
            "result": {
                "place_id": "ChIJtest",
                "name": "Test Place",
                "formatted_address": "Test St, Tirana",
                "geometry": {"location": {"lat": 41.3275, "lng": 19.8187}},
                "rating": 4.2,
                "user_ratings_total": 1500,
                "photos": [{"photo_reference": ref} for ref in (photo_refs or [])],
                "types": ["tourist_attraction"],
            },
        },
    )


def _photo_redirect(target: str) -> httpx.Response:
    return httpx.Response(302, headers={"location": target})


async def test_curated_hit_skips_cache_and_api(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    fake_db.queue(_curated_place())

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "curated"
    assert result.photo_url == "https://supabase.example/photos/berat-castle.jpg"
    assert result.rating == 4.7
    assert result.formatted_address == "Berat Castle, Berat"
    assert result.lat == 40.7038
    # Curated tier never touches the API or the cache table.
    assert mock_http.requests == []
    assert len(fake_db.executed) == 1   # only the curated SELECT


async def test_cache_hit_when_curated_misses(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    fake_db.queue(None)              # curated miss
    fake_db.queue(_cached_place())   # cache hit

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "cache"
    assert result.photo_url == "https://lh3.googleusercontent.example/cached.jpg"
    assert result.formatted_address == "Berat Castle, Berat, Albania"
    assert mock_http.requests == []


async def test_fresh_fetch_when_both_miss(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    fake_db.queue(None)   # curated miss
    fake_db.queue(None)   # cache miss

    mock_http.on(GOOGLE_DETAILS_URL, lambda req: _details_response(["ref1"]))
    mock_http.on(
        GOOGLE_PHOTO_URL,
        lambda req: _photo_redirect("https://lh3.googleusercontent.example/fresh.jpg"),
    )

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "fresh"
    assert result.photo_url == "https://lh3.googleusercontent.example/fresh.jpg"
    assert result.rating == 4.2
    assert result.lat == 41.3275
    # Two HTTP calls: details + one photo. Cache write was executed.
    assert len(mock_http.requests) == 2
    # The third execute is the upsert into places_cache.
    assert len(fake_db.executed) == 3


async def test_disabled_returns_none_when_curated_misses(
    fake_db, mock_http, monkeypatch
):
    _stub_settings(monkeypatch, places_enabled=False)
    fake_db.queue(None)   # curated miss

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is None
    assert mock_http.requests == []
    # Only the curated SELECT ran; the kill switch short-circuited before
    # the cache SELECT and any HTTP call.
    assert len(fake_db.executed) == 1


async def test_empty_api_key_treated_as_disabled(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch, key="")
    fake_db.queue(None)   # curated miss

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is None
    assert mock_http.requests == []


async def test_disabled_does_not_block_curated(fake_db, mock_http, monkeypatch):
    """Kill switch only gates cache + Google. Curated remains available."""
    _stub_settings(monkeypatch, places_enabled=False)
    fake_db.queue(_curated_place())

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "curated"


async def test_api_error_returns_none(fake_db, mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(GOOGLE_DETAILS_URL, lambda req: httpx.Response(500))

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is None


async def test_quota_exceeded_logs_clearly(
    fake_db, mock_http, monkeypatch, caplog
):
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(GOOGLE_DETAILS_URL, lambda req: httpx.Response(429))

    with caplog.at_level(logging.WARNING, logger="app.services.google_places"):
        result = await enrich_stop(fake_db, "ChIJtest")

    assert result is None
    assert any("quota" in rec.getMessage() for rec in caplog.records)


async def test_photo_resolution_failure_keeps_other_fields(
    fake_db, mock_http, monkeypatch
):
    _stub_settings(monkeypatch)
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(GOOGLE_DETAILS_URL, lambda req: _details_response(["ref1"]))
    # Photo endpoint fails — we should still cache + return the place,
    # just without a photo_url.
    mock_http.on(GOOGLE_PHOTO_URL, lambda req: httpx.Response(500))

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "fresh"
    assert result.photo_url is None
    assert result.rating == 4.2
    assert result.formatted_address == "Test St, Tirana"
    # Cache row was still written.
    assert len(fake_db.executed) == 3
