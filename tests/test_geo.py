"""Unit tests for `app.domain.trips.geo.is_in_albania`.

The bbox post-filter is the hard guard against the unbounded-text-search
failure (a geocoder query for an ambiguous English name resolving to
a US business). Cases here cover the four obvious neighbors (Kosovo,
North Macedonia, Greece, Sea) plus US coords with the wrong longitude
sign — the actual prod failure.
"""
from app.domain.trips.geo import is_in_albania


def test_inside_albania_typical_cities():
    assert is_in_albania(41.3275, 19.8187)   # Tirana
    assert is_in_albania(40.7038, 19.9457)   # Berat
    assert is_in_albania(40.0758, 20.1389)   # Gjirokastër
    assert is_in_albania(41.5094, 19.7931)   # Krujë


def test_inside_albania_near_edges():
    assert is_in_albania(39.65, 20.0)   # near Konispol (south edge, pad)
    assert is_in_albania(42.6, 20.0)    # near Vërmosh (north edge, pad)
    assert is_in_albania(40.0, 19.25)   # near Vlorë coast (west edge)
    assert is_in_albania(41.0, 21.05)   # near eastern border (east edge)


def test_outside_albania_neighboring_countries():
    assert not is_in_albania(37.98, 23.73)   # Athens (south of bbox)
    assert not is_in_albania(42.0, 21.43)    # Skopje (east of bbox)
    assert not is_in_albania(45.0, 20.5)     # Belgrade-ish (north of bbox)
    assert not is_in_albania(42.6, 21.16)    # Pristina (east of bbox)


def test_outside_albania_us_coords():
    # The actual production failure: an ambiguous English place name
    # resolved to a Brooklyn business. Negative longitude is the smoking
    # gun. Don't ever accept this.
    assert not is_in_albania(40.65, -73.95)   # Brooklyn
    assert not is_in_albania(42.33, -83.05)   # Detroit
    assert not is_in_albania(42.36, -71.06)   # Boston


def test_none_returns_false():
    assert not is_in_albania(None, None)
    assert not is_in_albania(41.3, None)
    assert not is_in_albania(None, 19.8)
