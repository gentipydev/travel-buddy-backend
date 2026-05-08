"""Unit tests for app.services.static_map.

Pure URL construction — no DB, no HTTP. The downsampler and the URL signer
are the load-bearing details; the markers/polylines paths are exercised
incidentally so the regression net is wide.
"""
from urllib.parse import unquote

from app.services.static_map import (
    build_static_map_url,
    decode_polyline,
    encode_polyline,
)


def test_url_includes_markers_per_day():
    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[
            [(40.5, 19.5), (40.6, 19.6)],
            [(41.0, 20.0)],
        ],
        polylines=[],
    )
    decoded = unquote(url)
    # Two days, so two markers= params with distinct colors.
    assert decoded.count("markers=") == 2
    assert "0xff5470" in decoded   # day 1 color
    assert "0xffa500" in decoded   # day 2 color
    assert "40.5,19.5" in decoded
    assert "40.6,19.6" in decoded
    assert "41.0,20.0" in decoded


def test_url_includes_polylines_with_distinct_colors():
    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=["abc", "def"],
    )
    decoded = unquote(url)
    assert decoded.count("path=") == 2
    assert "enc:abc" in decoded
    assert "enc:def" in decoded


def test_url_downsamples_when_too_long():
    # Construct a single very long polyline. The encoder produces ~5-7 bytes
    # per point, so 2000 points across [40, 41]×[19, 20] gives a polyline
    # that overshoots the 7800-char URL cap by a comfortable margin.
    points = [(40.0 + i * 0.0005, 19.0 + i * 0.0005) for i in range(2000)]
    enc = encode_polyline(points)
    assert len(enc) > 7900   # sanity: input is genuinely oversized

    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[enc],
    )
    assert len(url) < 7900   # downsampled or polyline-dropped to fit
    # Markers are never dropped by the downsampler.
    assert "40.5,19.5" in unquote(url)


def test_signing_produces_consistent_signature():
    secret = "dGVzdC1zZWNyZXQtMTIzNDU2"   # url-safe base64 nonsense
    url1 = build_static_map_url(
        api_key="k",
        signing_secret=secret,
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[],
    )
    url2 = build_static_map_url(
        api_key="k",
        signing_secret=secret,
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[],
    )
    assert "&signature=" in url1
    # Same input → same signature.
    assert url1 == url2
    # Different input → different signature.
    url3 = build_static_map_url(
        api_key="k",
        signing_secret=secret,
        markers_by_day=[[(40.6, 19.6)]],
        polylines=[],
    )
    sig1 = url1.split("&signature=")[1]
    sig3 = url3.split("&signature=")[1]
    assert sig1 != sig3


def test_unsigned_when_secret_missing():
    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[],
    )
    assert "signature=" not in url


def test_polyline_encode_decode_round_trips():
    # Sanity for the helper; the downsampler depends on this being correct.
    pts = [(40.5, 19.5), (40.6, 19.7), (40.55, 19.65)]
    enc = encode_polyline(pts)
    decoded = decode_polyline(enc)
    assert len(decoded) == len(pts)
    for (lat1, lng1), (lat2, lng2) in zip(pts, decoded):
        assert abs(lat1 - lat2) < 1e-4
        assert abs(lng1 - lng2) < 1e-4
