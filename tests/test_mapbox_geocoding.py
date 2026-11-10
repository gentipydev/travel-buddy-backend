"""Unit tests for app.integrations.mapbox_geocoding.

Covers the three-tier resolution (curated → cache → Mapbox), the Albania
bbox guard, the kill switch, and every failure path — all of which must
return None rather than raise, because trip saves run this best-effort.

HTTP is mocked via the `mock_http` fixture; the DB via `FakeSession`.
"""
import httpx
import pytest
from sqlalchemy.sql.dml import Insert

from app.integrations.mapbox_geocoding import (
    _FORWARD_URL,
    cache_key,
    resolve_place,
)

# The conftest auto-mark hook is broken under anyio 4.6 — without this every
# async test in the file silently skips.
pytestmark = pytest.mark.anyio


def _feature(
    lat: float,
    lng: float,
    *,
    name="Berat Castle",
    address="Berat, Albania",
    feature_type="poi",
):
    return {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [lng, lat]},
        "properties": {
            "name": name,
            "full_address": address,
            "feature_type": feature_type,
            "coordinates": {"latitude": lat, "longitude": lng},
        },
    }


def _upserts(db):
    """The cache-write statements the code under test issued (SELECTs excluded)."""
    return [stmt for stmt in db.executed if isinstance(stmt, Insert)]


def _ok(features):
    return lambda request: httpx.Response(
        200, json={"type": "FeatureCollection", "features": features}
    )


@pytest.fixture
def mapbox_on(monkeypatch):
    """Enable the integration with a dummy token."""
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "pk.test")
    monkeypatch.setattr(settings, "mapbox_geocoding_enabled", True)
    monkeypatch.setattr(settings, "mapbox_permanent_geocoding", False)
    return settings


# --- cache key -----------------------------------------------------------


def test_cache_key_is_accent_and_case_insensitive():
    # "Gjirokastër" and "gjirokaster" must land on the same cache row, or the
    # accented spelling Claude prefers would never hit a cached entry.
    assert cache_key("Gjirokastër", "Gjirokastër") == cache_key(
        "  gjirokaster ", "GJIROKASTER"
    )


def test_cache_key_separates_name_from_city():
    assert cache_key("Berat", None) == "berat|"
    assert cache_key("Berat", "Berat") == "berat|berat"


# --- tier 1: curated -----------------------------------------------------


async def test_curated_row_wins_and_supplies_photos_and_rating(fake_db):
    """The curated tier is the only source of photos/rating, and it must not
    make an HTTP call at all."""
    from app.db.models import Place

    fake_db.queue(
        Place(
            name="Berat",
            city="Berat",
            latitude=40.7058,
            longitude=19.9522,
            rating=4.8,
            photos=["https://supabase.example/berat-1.jpg"],
        )
    )
    result = await resolve_place(fake_db, name="Berat", city="Berat")

    assert result is not None
    assert result.source == "curated"
    assert (result.lat, result.lng) == (40.7058, 19.9522)
    assert result.rating == 4.8
    assert result.photos == ["https://supabase.example/berat-1.jpg"]
    assert result.formatted_address == "Berat, Berat"


async def test_curated_matches_across_accents(fake_db):
    from app.db.models import Place

    fake_db.queue(
        Place(name="Gjirokastër", city="Gjirokastër", latitude=40.07, longitude=20.14)
    )
    # The SQL comparison is case-insensitive; the Python re-filter handles the
    # accent. Query the unaccented spelling the model tends to emit.
    result = await resolve_place(fake_db, name="Gjirokastër", city="gjirokaster")
    assert result is not None and result.source == "curated"


async def test_curated_row_without_coords_falls_through(fake_db, mock_http, mapbox_on):
    """A curated row that can't place a marker must not short-circuit the
    geocoder — otherwise the stop silently loses its coordinates."""
    from app.db.models import Place

    fake_db.queue(Place(name="Berat", city="Berat", latitude=None, longitude=None))
    fake_db.queue(None)   # cache miss
    mock_http.on(_FORWARD_URL, _ok([_feature(40.7058, 19.9522)]))

    result = await resolve_place(fake_db, name="Berat", city="Berat")
    assert result is not None
    assert result.source == "fresh"


async def test_curated_tie_broken_by_city_then_rating(fake_db):
    from app.db.models import Place

    fake_db.queue([
        Place(name="Castle", city="Kruja", latitude=41.5, longitude=19.79, rating=4.1),
        Place(name="Castle", city="Berat", latitude=40.7, longitude=19.95, rating=4.9),
    ])
    result = await resolve_place(fake_db, name="Castle", city="Kruja")
    assert result is not None
    # City match wins even though the Berat row is rated higher.
    assert (result.lat, result.lng) == (41.5, 19.79)


# --- kill switch ---------------------------------------------------------


async def test_kill_switch_short_circuits_after_curated(fake_db, mock_http, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "pk.test")
    monkeypatch.setattr(settings, "mapbox_geocoding_enabled", False)

    fake_db.queue(None)   # curated miss
    result = await resolve_place(fake_db, name="Berat", city="Berat")

    assert result is None
    assert mock_http.requests == []   # never reached the network


async def test_missing_token_short_circuits(fake_db, mock_http, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "")
    fake_db.queue(None)

    assert await resolve_place(fake_db, name="Berat", city=None) is None
    assert mock_http.requests == []


# --- tier 2: cache -------------------------------------------------------


async def test_cache_hit_skips_the_network(fake_db, mock_http, mapbox_on):
    from app.db.models import GeocodeCache

    fake_db.queue(None)   # curated miss
    fake_db.queue(
        GeocodeCache(
            query_key="berat|berat",
            name="Berat",
            formatted_address="Berat, Albania",
            lat=40.7058,
            lng=19.9522,
        )
    )
    result = await resolve_place(fake_db, name="Berat", city="Berat")

    assert result is not None
    assert result.source == "cache"
    assert (result.lat, result.lng) == (40.7058, 19.9522)
    # Cache rows carry no photos or rating — those are curated-only.
    assert result.photos == []
    assert result.rating is None
    assert mock_http.requests == []


# --- tier 3: Mapbox ------------------------------------------------------


async def test_fresh_lookup_caches_and_scopes_the_query_to_albania(
    fake_db, mock_http, mapbox_on
):
    fake_db.queue(None)   # curated miss
    fake_db.queue(None)   # cache miss
    mock_http.on(_FORWARD_URL, _ok([_feature(40.7058, 19.9522)]))

    result = await resolve_place(fake_db, name="Berat Castle", city="Berat")

    assert result is not None
    assert result.source == "fresh"
    assert (result.lat, result.lng) == (40.7058, 19.9522)
    assert result.formatted_address == "Berat, Albania"

    request = mock_http.requests[0]
    params = request.url.params
    # ", Albania" is always appended so a bare city can't drift to a
    # same-named locality elsewhere.
    assert params["q"] == "Berat Castle, Berat, Albania"
    assert params["country"] == "al"
    assert params["limit"] == "1"
    assert "bbox" in params
    # `permanent` is only sent when the account licenses result storage.
    assert "permanent" not in params

    # The result was written to the cache under the normalized key.
    assert len(_upserts(fake_db)) == 1


async def test_permanent_flag_sent_when_enabled(fake_db, mock_http, mapbox_on, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_permanent_geocoding", True)
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(_FORWARD_URL, _ok([_feature(40.7058, 19.9522)]))

    await resolve_place(fake_db, name="Berat", city=None)
    assert mock_http.requests[0].url.params["permanent"] == "true"


async def test_geometry_coordinates_used_when_properties_omit_them(
    fake_db, mock_http, mapbox_on
):
    """Some feature types omit `properties.coordinates`; GeoJSON order is
    [lng, lat], so getting the fallback backwards would put every stop in
    the sea."""
    feature = _feature(40.7058, 19.9522)
    del feature["properties"]["coordinates"]
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(_FORWARD_URL, _ok([feature]))

    result = await resolve_place(fake_db, name="Berat", city=None)
    assert result is not None
    assert (result.lat, result.lng) == (40.7058, 19.9522)


async def test_out_of_bounds_result_is_dropped_and_not_cached(
    fake_db, mock_http, mapbox_on
):
    """Caching a non-AL row would poison every later lookup of the same name."""
    fake_db.queue(None)
    fake_db.queue(None)
    # Rome — plausibly returned for a confusable name, far outside Albania.
    mock_http.on(_FORWARD_URL, _ok([_feature(41.9028, 12.4964)]))

    assert await resolve_place(fake_db, name="Berat", city=None) is None
    assert not _upserts(fake_db), "an out-of-bounds hit must not be cached"


async def test_no_results_returns_none(fake_db, mock_http, mapbox_on):
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(_FORWARD_URL, _ok([]))

    assert await resolve_place(fake_db, name="Nowhere", city=None) is None


@pytest.mark.parametrize("status", [400, 401, 403, 404, 429, 500])
async def test_error_statuses_return_none(fake_db, mock_http, mapbox_on, status):
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        lambda request: httpx.Response(status, json={"message": "nope"}),
    )

    assert await resolve_place(fake_db, name="Berat", city=None) is None


async def test_timeout_returns_none(fake_db, mock_http, mapbox_on):
    def boom(request):
        raise httpx.TimeoutException("timed out", request=request)

    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(_FORWARD_URL, boom)

    assert await resolve_place(fake_db, name="Berat", city=None) is None


async def test_malformed_body_returns_none(fake_db, mock_http, mapbox_on):
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(_FORWARD_URL, lambda request: httpx.Response(200, text="not json"))

    assert await resolve_place(fake_db, name="Berat", city=None) is None


async def test_blank_name_never_matches_curated(fake_db, mock_http, monkeypatch):
    from app.config import settings

    monkeypatch.setattr(settings, "mapbox_access_token", "")
    # A whitespace-only name normalizes to "", which must not match a curated
    # row with an empty-ish name or hit the network.
    fake_db.queue(None)
    assert await resolve_place(fake_db, name="   ", city=None) is None
    assert mock_http.requests == []


# --- context fallback ----------------------------------------------------


async def test_unmatched_name_falling_back_to_the_city_is_dropped(
    fake_db, mock_http, mapbox_on
):
    """Forward search never returns empty for a query carrying a real city — it
    degrades to that city. Verified live: "Qwertyuiop Asdfgh Restaurant, Tirana"
    comes back as feature_type=place / name="Tirana". Accepting it would put a
    marker in central Tirana for a stop the model invented."""
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        _ok([_feature(41.3275, 19.8187, name="Tirana", feature_type="place")]),
    )

    result = await resolve_place(
        fake_db, name="Qwertyuiop Asdfgh Restaurant", city="Tirana"
    )
    assert result is None
    assert not _upserts(fake_db), "a context fallback must not be cached"


async def test_a_city_that_really_is_the_stop_still_resolves(
    fake_db, mock_http, mapbox_on
):
    """The mirror case: whole-day stops are often just a city. A coarse feature
    type is fine as long as the name is the one we asked for."""
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        _ok([_feature(40.7058, 19.9522, name="Berat", feature_type="place")]),
    )

    result = await resolve_place(fake_db, name="Berat", city="Berat")
    assert result is not None
    assert (result.lat, result.lng) == (40.7058, 19.9522)


async def test_city_name_match_ignores_accents(fake_db, mock_http, mapbox_on):
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        _ok([_feature(40.0758, 20.1389, name="Gjirokastër", feature_type="place")]),
    )

    result = await resolve_place(fake_db, name="Gjirokaster", city="Gjirokaster")
    assert result is not None


@pytest.mark.parametrize(
    "stop_name, city, returned_name",
    [
        # Verified live: Mapbox labels these differently from the query. Both
        # are the correct place, so a name-equality check would wrongly drop
        # them — the guard compares against the city instead.
        ("Sarandë", "Vlorë", "Saranda City"),
        ("Kruja", "Durrës", "Krujë"),
    ],
)
async def test_town_labelled_differently_from_the_query_is_kept(
    fake_db, mock_http, mapbox_on, stop_name, city, returned_name
):
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        _ok([_feature(40.0, 20.0, name=returned_name, feature_type="place")]),
    )

    result = await resolve_place(fake_db, name=stop_name, city=city)
    assert result is not None, f"{returned_name!r} is the right place, not a fallback"


async def test_coarse_result_without_a_city_query_is_kept(
    fake_db, mock_http, mapbox_on
):
    """With no city in the query there is no context to have fallen back to."""
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        _ok([_feature(41.3205, 19.8161, name="Tirana", feature_type="place")]),
    )

    assert await resolve_place(fake_db, name="Tirana", city=None) is not None


async def test_poi_result_is_never_treated_as_a_fallback(
    fake_db, mock_http, mapbox_on
):
    """A POI hit is authoritative even when its name differs from the query —
    "Bunk Art 2" vs "Bunk'Art 2" must not be rejected on punctuation."""
    fake_db.queue(None)
    fake_db.queue(None)
    mock_http.on(
        _FORWARD_URL,
        _ok([_feature(41.3270, 19.8196, name="Bunk'Art 2", feature_type="poi")]),
    )

    result = await resolve_place(fake_db, name="Bunk Art 2", city="Tirana")
    assert result is not None
    assert (result.lat, result.lng) == (41.3270, 19.8196)
