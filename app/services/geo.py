"""Geo helpers used by trip enrichment. Pure functions, stdlib only."""
from math import asin, cos, radians, sin, sqrt


_EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance between two lat/lng points in kilometers.

    Used by services.trip_enrichment to pick a travel mode heuristically:
    < 2km → walking, otherwise driving. The Directions API may decline
    walking if no walkable path exists; we accept whatever it returns.
    """
    lat1r, lat2r = radians(lat1), radians(lat2)
    dlat = lat2r - lat1r
    dlng = radians(lng2 - lng1)
    a = sin(dlat / 2) ** 2 + cos(lat1r) * cos(lat2r) * sin(dlng / 2) ** 2
    return 2 * _EARTH_RADIUS_KM * asin(sqrt(a))
