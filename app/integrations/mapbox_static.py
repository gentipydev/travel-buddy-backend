"""Build a Mapbox Static Images URL covering a trip's stops and routed
polylines. Pure URL construction, zero I/O.

The URL is persisted on `Trip.static_map_url` once at save/refine time so the
mobile client can show a thumbnail without re-rendering. It carries the public
`pk.` access token in the query string — the same token the app already ships,
so nothing secret is exposed. Restrict it by URL in the Mapbox account.
"""
from __future__ import annotations

from urllib.parse import quote

_BASE_URL = "https://api.mapbox.com/styles/v1/mapbox"
# Static Images renders classic styles only — Mapbox Standard (what
# TripMapScreen uses) is GL-only. `light-v11` is the closest match to the warm
# off-white AppColors palette. The two surfaces therefore don't render
# identically; that mismatch is deliberate and noted at both ends.
_DEFAULT_STYLE = "light-v11"
# Warm Mediterranean palette anchored on AppColors.accent1. Mirrors
# `kDayColors` in travel-buddy-mobile/lib/ui/screens/trip_map/trip_map_colors.dart.
# Repeats after 5 days. Mapbox wants bare 3- or 6-digit hex (no `#`, no `0x`).
_DAY_COLORS = (
    "e4935d",  # Day 1 — accent1 (brand warm orange)
    "a8453e",  # Day 2 — terracotta
    "c9a55a",  # Day 3 — warm gold
    "6b8e5c",  # Day 4 — olive sage
    "5a7a8e",  # Day 5 — Adriatic slate
)
_PATH_WIDTH = 4
_PATH_OPACITY = "0.9"
_MAX_URL_LEN = 7800   # margin under Mapbox's 8192 hard cap


def _color_for(day_index: int) -> str:
    return _DAY_COLORS[day_index % len(_DAY_COLORS)]


def _label_for_anchor(day_index: int) -> str | None:
    """Single-char label for a day's anchor marker. `pin-s` accepts a digit or
    a lowercase letter; we use 1-9 then a-z, dropping the label past day 35
    (only matters in theory — real trips don't exceed a handful of days).

    One anchor per day at the day's centroid: the thumbnail shows a clean
    "5 dots for 5 days" instead of 15-20 per-stop pins that overflow the
    ~220 dp wide chat bubble. The full per-stop detail lives on
    `TripMapScreen`.
    """
    if day_index < 9:
        return str(day_index + 1)
    if day_index < 35:
        return chr(ord("a") + (day_index - 9))
    return None


def encode_polyline(points: list[tuple[float, float]]) -> str:
    """The precision-5 polyline algorithm: signed deltas in 1e5 units, 5-bit
    chunks with the high bit signaling continuation. Mirrors `decode_polyline`.
    This is what Mapbox returns for `geometries=polyline`.
    """
    out: list[str] = []
    prev_lat = 0
    prev_lng = 0
    for lat, lng in points:
        ilat = round(lat * 1e5)
        ilng = round(lng * 1e5)
        out.append(_encode_signed(ilat - prev_lat))
        out.append(_encode_signed(ilng - prev_lng))
        prev_lat = ilat
        prev_lng = ilng
    return "".join(out)


def decode_polyline(encoded: str) -> list[tuple[float, float]]:
    """Inverse of `encode_polyline`. Used by the downsampler."""
    points: list[tuple[float, float]] = []
    i = 0
    lat = 0
    lng = 0
    while i < len(encoded):
        dlat, i = _decode_signed(encoded, i)
        dlng, i = _decode_signed(encoded, i)
        lat += dlat
        lng += dlng
        points.append((lat / 1e5, lng / 1e5))
    return points


def _encode_signed(value: int) -> str:
    v = value << 1
    if value < 0:
        v = ~v
    chunks: list[str] = []
    while v >= 0x20:
        chunks.append(chr((0x20 | (v & 0x1F)) + 63))
        v >>= 5
    chunks.append(chr(v + 63))
    return "".join(chunks)


def _decode_signed(encoded: str, i: int) -> tuple[int, int]:
    result = 0
    shift = 0
    while True:
        b = ord(encoded[i]) - 63
        i += 1
        result |= (b & 0x1F) << shift
        shift += 5
        if b < 0x20:
            break
    delta = ~(result >> 1) if (result & 1) else (result >> 1)
    return delta, i


def _downsample(encoded: str, stride: int) -> str:
    pts = decode_polyline(encoded)
    if len(pts) <= 2:
        return encoded
    # Always keep first and last so the line still spans endpoints.
    kept = [pts[0]] + pts[1:-1:stride] + [pts[-1]]
    return encode_polyline(kept)


def build_static_map_url(
    *,
    access_token: str,
    width: int = 600,
    height: int = 300,
    markers_by_day: list[list[tuple[float, float]]],
    polylines: list[str],
    style: str = _DEFAULT_STYLE,
) -> str:
    """Assemble a Static Images URL.

    `markers_by_day[i]` is the ordered list of (lat, lng) for day i.
    `polylines[j]` is one encoded polyline; colored to match its day if
    available (we associate by index — caller passes them in day order).

    Overlays are drawn in path-then-pin order so markers sit on top of the
    route. The viewport is `auto`, which fits every overlay — no centre or
    zoom math needed.

    If the constructed URL exceeds `_MAX_URL_LEN`, polylines are
    progressively downsampled (stride 2, 3, …) until under cap. Markers
    are never dropped — they carry the most semantic value.
    """
    polylines = list(polylines)   # local copy; downsampler mutates
    stride = 1
    while True:
        overlays: list[str] = []
        for poly_index, enc in enumerate(polylines):
            color = _color_for(poly_index)
            # The polyline is encoded as a URI *component*: its alphabet
            # includes `\`, `?`, `|` and `@`, any of which would otherwise
            # terminate or corrupt the overlay segment.
            overlays.append(
                f"path-{_PATH_WIDTH}+{color}-{_PATH_OPACITY}"
                f"({quote(enc, safe='')})"
            )
        # One pin per day at the day's centroid (mean lat/lng of its stops).
        # At thumbnail size the per-stop labels are unreadable; the per-day
        # anchor + day-colored polyline reads as "N-day trip" while keeping
        # the URL short. Per-stop detail lives on the interactive
        # TripMapScreen.
        for day_index, points in enumerate(markers_by_day):
            if not points:
                continue
            color = _color_for(day_index)
            centroid_lat = sum(lat for lat, _ in points) / len(points)
            centroid_lng = sum(lng for _, lng in points) / len(points)
            label = _label_for_anchor(day_index)
            name = f"pin-s-{label}" if label is not None else "pin-s"
            # Mapbox takes the marker anchor longitude-first.
            overlays.append(
                f"{name}+{color}({centroid_lng:.5f},{centroid_lat:.5f})"
            )

        overlay = ",".join(overlays)
        path = f"{_BASE_URL}/{style}/static"
        if overlay:
            path += f"/{overlay}"
        url = f"{path}/auto/{width}x{height}?access_token={quote(access_token)}"

        if len(url) <= _MAX_URL_LEN or not polylines:
            break
        stride += 1
        polylines = [_downsample(p, stride) for p in polylines]
        # Bail out at extreme stride — at this point dropping all polylines
        # is the only thing that can shrink the URL further.
        if stride > 16:
            polylines = []

    return url
