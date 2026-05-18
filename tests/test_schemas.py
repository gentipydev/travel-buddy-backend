"""Unit tests for the coord-stripping validators on TripStopCreate /
TripStopUpdate.

The Anthropic tool schema allows the model to emit raw lat/lng with no
bounds — these validators are the parse-boundary guard against
US-coords leaking into a saved Albanian trip.
"""
from app.schemas import TripStopCreate, TripStopUpdate


def test_stop_create_strips_us_coords():
    stop = TripStopCreate(name="X", kind="sight", lat=40.65, lng=-73.95)
    assert stop.lat is None
    assert stop.lng is None
    # The rest of the stop is left intact — enrichment can re-resolve.
    assert stop.name == "X"
    assert stop.kind == "sight"


def test_stop_create_keeps_albania_coords():
    stop = TripStopCreate(name="X", kind="sight", lat=41.3275, lng=19.8187)
    assert stop.lat == 41.3275
    assert stop.lng == 19.8187


def test_stop_create_strips_half_coords_lat_only():
    stop = TripStopCreate(name="X", kind="sight", lat=41.3, lng=None)
    assert stop.lat is None
    assert stop.lng is None


def test_stop_create_strips_half_coords_lng_only():
    stop = TripStopCreate(name="X", kind="sight", lat=None, lng=19.8)
    assert stop.lat is None
    assert stop.lng is None


def test_stop_create_no_coords_unchanged():
    stop = TripStopCreate(name="X", kind="sight")
    assert stop.lat is None
    assert stop.lng is None


def test_stop_update_strips_us_coords():
    patch = TripStopUpdate(lat=40.65, lng=-73.95)
    assert patch.lat is None
    assert patch.lng is None


def test_stop_update_keeps_albania_coords():
    patch = TripStopUpdate(lat=41.3275, lng=19.8187)
    assert patch.lat == 41.3275
    assert patch.lng == 19.8187


def test_stop_update_strips_half_coords():
    patch = TripStopUpdate(lat=41.3, lng=None)
    assert patch.lat is None
    assert patch.lng is None
