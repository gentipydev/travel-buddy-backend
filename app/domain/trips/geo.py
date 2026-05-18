"""Geo helpers used by trip enrichment. Pure functions, stdlib only."""
from math import asin, cos, radians, sin, sqrt


_EARTH_RADIUS_KM = 6371.0088


# Padded slightly past Albania's official extent (39.624–42.665 lat /
# 19.286–21.057 lng) so coordinate jitter doesn't drop legitimate points
# without admitting any neighboring-country territory.
ALBANIA_LAT_MIN = 39.5
ALBANIA_LAT_MAX = 42.7
ALBANIA_LNG_MIN = 19.2
ALBANIA_LNG_MAX = 21.1


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points in kilometers.

    Used by trip enrichment to pick a travel mode heuristically: < 2km →
    walking, otherwise driving. The Directions API may decline walking if
    no walkable path exists; we accept whatever it returns.
    """
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlat = lat2r - lat1r
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlng / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * asin(sqrt(a))


def is_in_albania(lat: float | None, lng: float | None) -> bool:
    """True iff (lat, lng) is inside Albania's padded bounding box.

    A null pair (or half-null) returns False — callers should treat
    "no coords" as a separate state from "coords outside Albania."
    """
    if lat is None or lng is None:
        return False
    return (
        ALBANIA_LAT_MIN <= lat <= ALBANIA_LAT_MAX
        and ALBANIA_LNG_MIN <= lng <= ALBANIA_LNG_MAX
    )
