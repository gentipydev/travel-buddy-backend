"""Build a signed Google Static Maps URL covering a trip's stops and routed
polylines. Pure URL construction, zero I/O.

The URL is persisted on `Trip.static_map_url` once at save/refine time so the
mobile client can show a thumbnail without re-rendering. Signing makes the
URL safe to embed publicly (per Google's URL-Signing-Secret docs).
"""
from __future__ import annotations

import base64
import hashlib
import hmac
from urllib.parse import quote


_BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"
# Warm Mediterranean palette anchored on AppColors.accent1. Mirrors
# `kDayColors` in travel-buddy-mobile/lib/ui/screens/trip_map/trip_map_colors.dart.
# Repeats after 5 days.
_DAY_COLORS = (
    "0xe4935d",  # Day 1 — accent1 (brand warm orange)
    "0xa8453e",  # Day 2 — terracotta
    "0xc9a55a",  # Day 3 — warm gold
    "0x6b8e5c",  # Day 4 — olive sage
    "0x5a7a8e",  # Day 5 — Adriatic slate
)
_MAX_URL_LEN = 7800   # margin under Google's ~8192 hard cap


def _color_for(day_index: int) -> str:
    return _DAY_COLORS[day_index % len(_DAY_COLORS)]


def _label_for_stop(stop_index: int) -> str | None:
    """Single-char label for a stop's position in the trip's overall stop
    order. Static Maps allows {0-9, A-Z}; we use 1-9 then A-Z, dropping
    the label past stop 35 (the visual is fine without it for very long trips).

    Per-stop (not per-day) labeling: every marker gets its own unique label,
    so two stops never share a digit. Day grouping comes from `_color_for`.
    """
    if stop_index < 9:
        return str(stop_index + 1)
    if stop_index < 35:
        return chr(ord("A") + (stop_index - 9))
    return None


def encode_polyline(points: list[tuple[float, float]]) -> str:
    """Google's polyline algorithm: signed deltas in 1e5 units, 5-bit chunks
    with the high bit signaling continuation. Mirrors `decode_polyline`.
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
    api_key: str,
    signing_secret: str,
    width: int = 600,
    height: int = 300,
    markers_by_day: list[list[tuple[float, float]]],
    polylines: list[str],
    map_type: str = "roadmap",
) -> str:
    """Assemble a signed Static Maps URL.

    `markers_by_day[i]` is the ordered list of (lat, lng) for day i.
    `polylines[j]` is one encoded polyline; colored to match its day if
    available (we associate by index — caller passes them in day order).

    If the constructed URL exceeds `_MAX_URL_LEN`, polylines are
    progressively downsampled (stride 2, 3, …) until under cap. Markers
    are never dropped — they carry the most semantic value.
    """
    polylines = list(polylines)   # local copy; downsampler mutates
    stride = 1
    while True:
        params: list[str] = [
            f"size={width}x{height}",
            f"maptype={quote(map_type)}",
        ]
        # One markers= param per stop — each gets a unique single-char label
        # in trip order (1..9 then A..Z; unlabeled past 35). Day color is
        # still applied per day so the visual grouping is preserved.
        stop_index = 0
        for day_index, points in enumerate(markers_by_day):
            color = _color_for(day_index)
            for lat, lng in points:
                label = _label_for_stop(stop_index)
                spec = [f"color:{color}"]
                if label is not None:
                    spec.append(f"label:{label}")
                spec.append(f"{lat},{lng}")
                # Encode `|` as %7C: clients (e.g. Dart's Image.network) re-encode
                # `|` before transmission. Signing the encoded form keeps the bytes
                # the server signs identical to the bytes Google receives, so the
                # HMAC verifies. Leaving `|` literal here causes a 403 from Google.
                params.append("markers=" + quote("|".join(spec), safe=":,"))
                stop_index += 1
        for poly_index, enc in enumerate(polylines):
            color = _color_for(poly_index)
            params.append(
                f"path=" + quote(
                    f"enc:{enc}|color:{color}|weight:4",
                    safe=":,",
                )
            )
        if api_key:
            params.append(f"key={quote(api_key)}")
        path_query = "/maps/api/staticmap?" + "&".join(params)
        url = "https://maps.googleapis.com" + path_query

        if len(url) <= _MAX_URL_LEN or not polylines:
            break
        stride += 1
        polylines = [_downsample(p, stride) for p in polylines]
        # Bail out at extreme stride — at this point dropping all polylines
        # is the only thing that can shrink the URL further.
        if stride > 16:
            polylines = []

    if signing_secret:
        url = _sign_url(url, signing_secret)
    return url


def _sign_url(url: str, signing_secret: str) -> str:
    """HMAC-SHA1 of the path+query, keyed by the URL-safe-base64-decoded
    secret, encoded back as URL-safe base64. Matches Google's docs:
    https://developers.google.com/maps/documentation/maps-static/digital-signature
    """
    # Strip the host: signing operates on path + query only.
    prefix, _, path_query = url.partition("googleapis.com")
    host_url = prefix + "googleapis.com"
    decoded_key = base64.urlsafe_b64decode(signing_secret)
    signature = base64.urlsafe_b64encode(
        hmac.new(decoded_key, path_query.encode("utf-8"), hashlib.sha1).digest()
    ).decode("ascii")
    return f"{host_url}{path_query}&signature={signature}"
