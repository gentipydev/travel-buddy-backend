"""Unit tests for app.integrations.google_places.enrich_stop.

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
from app.integrations import google_places
from app.integrations.google_places import enrich_stop, find_place_id_by_name


GOOGLE_DETAILS_URL = "https://maps.googleapis.com/maps/api/place/details/json"
GOOGLE_PHOTO_URL = "https://maps.googleapis.com/maps/api/place/photo"
GOOGLE_TEXTSEARCH_URL = (
    "https://maps.googleapis.com/maps/api/place/textsearch/json"
)


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
    assert result.additional_photos == []
    assert result.rating == 4.2
    assert result.formatted_address == "Test St, Tirana"
    # Cache row was still written.
    assert len(fake_db.executed) == 3


# -- Multi-photo (added with migration 014) -------------------------------


async def test_curated_returns_all_photos_not_just_primary(
    fake_db, mock_http, monkeypatch
):
    """Curated rows can carry multiple photos in the JSONB column. The
    enrichment result must surface them as `additional_photos` so the
    `_StopPhotoStack` UI can render hero + thumbs."""
    _stub_settings(monkeypatch)
    place = _curated_place()
    place.photos = [
        "https://supabase.example/p1.jpg",
        "https://supabase.example/p2.jpg",
        "https://supabase.example/p3.jpg",
    ]
    fake_db.queue(place)

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "curated"
    assert result.photo_url == "https://supabase.example/p1.jpg"
    assert result.additional_photos == [
        "https://supabase.example/p2.jpg",
        "https://supabase.example/p3.jpg",
    ]


async def test_cache_returns_additional_photos_from_jsonb(
    fake_db, mock_http, monkeypatch
):
    """Cache rows persist `additional_photo_urls` from the original Google
    fetch (places_cache.additional_photo_urls JSONB). The hot path reads
    them straight off the row — no second Google call."""
    _stub_settings(monkeypatch)
    cached = _cached_place()
    cached.additional_photo_urls = [
        "https://lh3.googleusercontent.example/cached2.jpg",
        "https://lh3.googleusercontent.example/cached3.jpg",
    ]
    fake_db.queue(None)      # curated miss
    fake_db.queue(cached)    # cache hit

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "cache"
    assert result.photo_url == "https://lh3.googleusercontent.example/cached.jpg"
    assert result.additional_photos == [
        "https://lh3.googleusercontent.example/cached2.jpg",
        "https://lh3.googleusercontent.example/cached3.jpg",
    ]
    # No Google call for cache hits.
    assert mock_http.requests == []


async def test_fresh_resolves_multiple_photo_refs(
    fake_db, mock_http, monkeypatch
):
    """When Google Place Details returns multiple `photo_reference`s,
    `_resolve_photos` redirects each through `/place/photo` and the
    enrichment result returns the first as `photo_url` and the rest as
    `additional_photos`."""
    _stub_settings(monkeypatch)
    fake_db.queue(None)   # curated miss
    fake_db.queue(None)   # cache miss

    mock_http.on(
        GOOGLE_DETAILS_URL,
        lambda req: _details_response(["refA", "refB", "refC"]),
    )
    # Each photo redirect resolves to a unique target — order matches the
    # order of the photo_reference list (Google preserves it).
    photo_targets = iter([
        "https://lh3.googleusercontent.example/A.jpg",
        "https://lh3.googleusercontent.example/B.jpg",
        "https://lh3.googleusercontent.example/C.jpg",
    ])
    mock_http.on(
        GOOGLE_PHOTO_URL,
        lambda req: _photo_redirect(next(photo_targets)),
    )

    result = await enrich_stop(fake_db, "ChIJtest")

    assert result is not None
    assert result.source == "fresh"
    assert result.photo_url == "https://lh3.googleusercontent.example/A.jpg"
    assert result.additional_photos == [
        "https://lh3.googleusercontent.example/B.jpg",
        "https://lh3.googleusercontent.example/C.jpg",
    ]
    # 1 details call + 3 photo redirects = 4 HTTP calls.
    assert len(mock_http.requests) == 4


def _textsearch_response(
    place_id: str = "ChIJsk_text",
    lat: float = 41.3275,
    lng: float = 19.8187,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "status": "OK",
            "results": [
                {
                    "place_id": place_id,
                    "name": "Skanderbeg Square",
                    "formatted_address": "Skanderbeg Square, Tirana, Albania",
                    "geometry": {"location": {"lat": lat, "lng": lng}},
                }
            ],
        },
    )


async def test_textsearch_happy_path(mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    mock_http.on(GOOGLE_TEXTSEARCH_URL, lambda req: _textsearch_response())

    result = await find_place_id_by_name(
        name="Skanderbeg Square", city="Tirana"
    )

    assert result == ("ChIJsk_text", 41.3275, 19.8187)
    # Query string combines name + city + "Albania" so an ambiguous English
    # name can't drift to a same-named locality elsewhere. `region=al` is a
    # ranking bias; the bbox post-filter is the hard guard.
    url = str(mock_http.requests[0].url)
    assert "Skanderbeg%20Square%2C%20Tirana%2C%20Albania" in url
    assert "region=al" in url


async def test_textsearch_no_city_still_includes_albania(mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    mock_http.on(GOOGLE_TEXTSEARCH_URL, lambda req: _textsearch_response())

    result = await find_place_id_by_name(name="Skanderbeg Square")

    assert result == ("ChIJsk_text", 41.3275, 19.8187)
    url = str(mock_http.requests[0].url)
    assert "Skanderbeg%20Square%2C%20Albania" in url
    assert "region=al" in url


async def test_textsearch_drops_out_of_bounds_result(
    mock_http, monkeypatch, caplog
):
    """The smoking-gun failure: Google returns a US business for an
    ambiguous English name. The bbox post-filter must refuse to write
    those coords onto the stop, regardless of the name+region bias."""
    _stub_settings(monkeypatch)
    # Brooklyn coords (the actual production failure pattern).
    mock_http.on(
        GOOGLE_TEXTSEARCH_URL,
        lambda req: _textsearch_response(
            place_id="ChIJbrooklyn", lat=40.65, lng=-73.95
        ),
    )

    with caplog.at_level(logging.WARNING, logger="app.services.google_places"):
        result = await find_place_id_by_name(
            name="Mediterranean Cafe", city="Tirana"
        )

    assert result is None
    assert any(
        "dropped=out_of_bounds" in rec.getMessage() for rec in caplog.records
    )


async def test_fetch_and_cache_drops_out_of_bounds_geometry(
    fake_db, mock_http, monkeypatch
):
    """If a Place Details response has a non-AL geometry (e.g. Claude
    emitted a bogus place_id that points outside Albania), we must not
    seed the cache with that row — a future caller with the same id
    would still pull the wrong country."""
    _stub_settings(monkeypatch)
    fake_db.queue(None)   # curated miss
    fake_db.queue(None)   # cache miss

    def _us_details(req):
        return httpx.Response(
            200,
            json={
                "status": "OK",
                "result": {
                    "place_id": "ChIJus",
                    "name": "Tirana Restaurant",
                    "formatted_address": "123 Main St, Brooklyn, NY",
                    "geometry": {"location": {"lat": 40.65, "lng": -73.95}},
                    "rating": 4.4,
                    "user_ratings_total": 200,
                    "photos": [],
                    "types": ["restaurant"],
                },
            },
        )

    mock_http.on(GOOGLE_DETAILS_URL, _us_details)

    result = await enrich_stop(fake_db, "ChIJus")

    assert result is None
    # Exactly two executes ran (curated + cache SELECT). The cache INSERT
    # must NOT have run — otherwise the bad row would poison future hits.
    assert len(fake_db.executed) == 2


async def test_textsearch_zero_results_returns_none(mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    mock_http.on(
        GOOGLE_TEXTSEARCH_URL,
        lambda req: httpx.Response(
            200, json={"status": "ZERO_RESULTS", "results": []}
        ),
    )

    result = await find_place_id_by_name(name="Nonexistent Place")

    assert result is None


async def test_textsearch_kill_switch_skips_http(mock_http, monkeypatch):
    _stub_settings(monkeypatch, places_enabled=False)

    result = await find_place_id_by_name(
        name="Skanderbeg Square", city="Tirana"
    )

    assert result is None
    assert mock_http.requests == []


async def test_textsearch_empty_api_key_skips_http(mock_http, monkeypatch):
    _stub_settings(monkeypatch, key="")

    result = await find_place_id_by_name(name="Skanderbeg Square")

    assert result is None
    assert mock_http.requests == []


async def test_textsearch_http_error_returns_none(mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    mock_http.on(GOOGLE_TEXTSEARCH_URL, lambda req: httpx.Response(500))

    result = await find_place_id_by_name(name="Skanderbeg Square")

    assert result is None


async def test_textsearch_drops_result_without_geometry(mock_http, monkeypatch):
    _stub_settings(monkeypatch)
    mock_http.on(
        GOOGLE_TEXTSEARCH_URL,
        lambda req: httpx.Response(
            200,
            json={
                "status": "OK",
                "results": [{"place_id": "ChIJnogeo", "name": "x"}],
            },
        ),
    )

    result = await find_place_id_by_name(name="X")

    assert result is None
