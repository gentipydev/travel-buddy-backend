"""Unit tests for app.services.static_map.

Pure URL construction — no DB, no HTTP. The downsampler and the URL signer
are the load-bearing details; the markers/polylines paths are exercised
incidentally so the regression net is wide.
"""
from urllib.parse import unquote

from app.services.static_map import (
    _label_for_stop,
    build_static_map_url,
    decode_polyline,
    encode_polyline,
)


def test_url_emits_one_marker_per_stop_with_unique_labels():
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
    # 3 stops total → 3 markers= params with unique labels 1, 2, 3.
    assert decoded.count("markers=") == 3
    # Day color still applied per day; day 1's two stops share the brand
    # warm-orange (accent1), day 2 is terracotta.
    assert decoded.count("0xe4935d") == 2
    assert "0xa8453e" in decoded
    # Each stop gets a unique label in trip order.
    assert "label:1" in decoded
    assert "label:2" in decoded
    assert "label:3" in decoded
    assert "40.5,19.5" in decoded
    assert "40.6,19.6" in decoded
    assert "41.0,20.0" in decoded


def test_label_for_stop_alphabet_rule():
    # 1..9 for the first nine stops.
    assert [_label_for_stop(i) for i in range(9)] == list("123456789")
    # A..Z for the next 26 stops.
    assert [_label_for_stop(9 + i) for i in range(26)] == list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    # Unlabeled past stop 35 — the visual is fine without it.
    assert _label_for_stop(35) is None
    assert _label_for_stop(100) is None


def test_url_omits_label_past_stop_35():
    # 36 stops in one day; the 36th stop should be unlabeled.
    points = [(40.0 + i * 0.001, 19.0) for i in range(36)]
    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[points],
        polylines=[],
    )
    decoded = unquote(url)
    assert decoded.count("markers=") == 36
    # Last stop (index 35): coords present but no label embedded for it.
    assert "label:Z" in decoded
    last_marker_segment = decoded.rsplit("markers=", 1)[-1]
    assert "label:" not in last_marker_segment
    assert "40.035,19.0" in last_marker_segment


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
