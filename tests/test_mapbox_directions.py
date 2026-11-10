"""Unit tests for app.integrations.mapbox_directions.

Covers the cache key, the cache tier, the kill switch, and every failure
path. `get_route` runs best-effort inside trip enrichment, so nothing here
may raise — a miss just means the client draws a straight connector.

HTTP is mocked via the `mock_http` fixture; the DB via `FakeSession`.
"""
import httpx
import pytest

from app.integrations.mapbox_directions import (
    _DIRECTIONS_URL,
    _cache_key,
    get_route,
)

# The conftest auto-mark hook is broken under anyio 4.6 — without this every
# async test in the file silently skips.
pytestmark = pytest.mark.anyio


# Berat → Gjirokastër, roughly.
FROM = (40.7058, 19.9522)
TO = (40.0758, 20.1389)


def _route_response(geometry="_p~iF~ps|U_ulLnnqC", distance=1234.7, duration=567.2):
    return lambda request: httpx.Response(
        200,
        json={
            "code": "Ok",
            "routes": [
                {
                    "geometry": geometry,
                    "distance": distance,
                    "duration": duration,
                }
            ],
        },
    )


@pytest.fixture
def mapbox_on(monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "pk.test")
    monkeypatch.setattr(settings, "mapbox_directions_enabled", True)
    return settings


# --- cache key -----------------------------------------------------------


def test_cache_key_rounds_to_about_ten_metres():
    """Two stops geocoded a few metres apart must share a cached polyline."""
    a = _cache_key(40.70581, 19.95221, 40.07581, 20.13891, "driving")
    b = _cache_key(40.70579, 19.95219, 40.07579, 20.13889, "driving")
    assert a == b


def test_cache_key_separates_travel_modes():
    assert _cache_key(*FROM, *TO, "driving") != _cache_key(*FROM, *TO, "walking")


def test_cache_key_is_directional():
    assert _cache_key(*FROM, *TO, "driving") != _cache_key(*TO, *FROM, "driving")


# --- kill switch ---------------------------------------------------------


async def test_kill_switch_short_circuits(fake_db, mock_http, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "pk.test")
    monkeypatch.setattr(settings, "mapbox_directions_enabled", False)

    assert await get_route(fake_db, *FROM, *TO) is None
    assert mock_http.requests == []


async def test_missing_token_short_circuits(fake_db, mock_http, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "")

    assert await get_route(fake_db, *FROM, *TO) is None
    assert mock_http.requests == []


# --- cache tier ----------------------------------------------------------


async def test_cache_hit_skips_the_network(fake_db, mock_http, mapbox_on):
    from app.db.models import DirectionsCache

    cached = DirectionsCache(
        cache_key=_cache_key(*FROM, *TO, "driving"),
        encoded_polyline="abc",
        distance_meters=100,
        duration_seconds=60,
        travel_mode="driving",
    )
    fake_db.queue(cached)

    result = await get_route(fake_db, *FROM, *TO)
    assert result is cached
    assert mock_http.requests == []


# --- fresh fetch ---------------------------------------------------------


async def test_fresh_fetch_sends_lng_first_coords_and_caches(
    fake_db, mock_http, mapbox_on
):
    fake_db.queue(None)   # cache miss
    mock_http.on(_DIRECTIONS_URL, _route_response())
    # The upsert re-reads the row; queue what the SELECT should return.
    from app.db.models import DirectionsCache

    fake_db.queue(DirectionsCache(cache_key="k", encoded_polyline="x", travel_mode="driving"))

    result = await get_route(fake_db, *FROM, *TO, "driving")
    assert result is not None

    request = mock_http.requests[0]
    # Mapbox takes longitude first, semicolon-separated — swapping these puts
    # every route in the wrong hemisphere.
    assert str(request.url).startswith(
        f"{_DIRECTIONS_URL}/driving/19.9522,40.7058;20.1389,40.0758"
    )
    params = request.url.params
    # polyline (precision 5) is what the DB column and the Flutter decoder
    # expect; polyline6 or geojson would break both.
    assert params["geometries"] == "polyline"
    assert params["overview"] == "full"
    assert params["access_token"] == "pk.test"
    assert params["radiuses"] == "250;250"


async def test_walking_profile_is_used_verbatim(fake_db, mock_http, mapbox_on):
    from app.db.models import DirectionsCache

    fake_db.queue(None)
    mock_http.on(_DIRECTIONS_URL, _route_response())
    fake_db.queue(DirectionsCache(cache_key="k", encoded_polyline="x", travel_mode="walking"))

    await get_route(fake_db, *FROM, *TO, "walking")
    assert f"{_DIRECTIONS_URL}/walking/" in str(mock_http.requests[0].url)


async def test_float_distance_and_duration_are_rounded_to_int(
    fake_db, mock_http, mapbox_on
):
    """Mapbox reports metres/seconds as floats; the columns are integers."""
    from sqlalchemy.sql.dml import Insert

    from app.db.models import DirectionsCache

    fake_db.queue(None)
    mock_http.on(_DIRECTIONS_URL, _route_response(distance=1234.7, duration=567.2))
    fake_db.queue(DirectionsCache(cache_key="k", encoded_polyline="x", travel_mode="driving"))

    await get_route(fake_db, *FROM, *TO)

    insert = next(s for s in fake_db.executed if isinstance(s, Insert))
    values = insert.compile().params
    assert values["distance_meters"] == 1235
    assert values["duration_seconds"] == 567
    assert isinstance(values["distance_meters"], int)


# --- failure paths -------------------------------------------------------


@pytest.mark.parametrize("code", ["NoRoute", "NoSegment", "InvalidInput"])
async def test_non_ok_api_code_is_not_cached(fake_db, mock_http, mapbox_on, code):
    """These come back HTTP 200. Caching them would pin a permanent miss."""
    from sqlalchemy.sql.dml import Insert

    fake_db.queue(None)
    mock_http.on(
        _DIRECTIONS_URL,
        lambda request: httpx.Response(200, json={"code": code, "routes": []}),
    )

    assert await get_route(fake_db, *FROM, *TO) is None
    assert not [s for s in fake_db.executed if isinstance(s, Insert)]


async def test_route_without_geometry_returns_none(fake_db, mock_http, mapbox_on):
    fake_db.queue(None)
    mock_http.on(
        _DIRECTIONS_URL,
        lambda request: httpx.Response(
            200, json={"code": "Ok", "routes": [{"distance": 10, "duration": 5}]}
        ),
    )
    assert await get_route(fake_db, *FROM, *TO) is None


@pytest.mark.parametrize("status", [400, 401, 403, 422, 429, 500])
async def test_error_statuses_return_none(fake_db, mock_http, mapbox_on, status):
    fake_db.queue(None)
    mock_http.on(
        _DIRECTIONS_URL,
        lambda request: httpx.Response(status, json={"message": "nope"}),
    )
    assert await get_route(fake_db, *FROM, *TO) is None


async def test_timeout_returns_none(fake_db, mock_http, mapbox_on):
    def boom(request):
        raise httpx.TimeoutException("timed out", request=request)

    fake_db.queue(None)
    mock_http.on(_DIRECTIONS_URL, boom)
    assert await get_route(fake_db, *FROM, *TO) is None


async def test_malformed_body_returns_none(fake_db, mock_http, mapbox_on):
    fake_db.queue(None)
    mock_http.on(_DIRECTIONS_URL, lambda request: httpx.Response(200, text="not json"))
    assert await get_route(fake_db, *FROM, *TO) is None
