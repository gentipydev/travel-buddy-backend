"""Unit tests for app.integrations.static_maps.

Pure URL construction — no DB, no HTTP. The downsampler and the URL signer
are the load-bearing details; the markers/polylines paths are exercised
incidentally so the regression net is wide.
"""
from urllib.parse import unquote

from app.integrations.static_maps import (
    _label_for_anchor,
    build_static_map_url,
    decode_polyline,
    encode_polyline,
)


def test_url_emits_one_marker_per_day_at_centroid():
    # Day 1: two stops, centroid (40.55, 19.55). Day 2: one stop (its own pos).
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
    # Two non-empty days → exactly two markers= params (down from per-stop = 3).
    assert decoded.count("markers=") == 2
    # Day 1 centroid coords; day 2 single-stop position.
    assert "40.55,19.55" in decoded
    assert "41.0,20.0" in decoded
    # Per-day colors preserved.
    assert "0xe4935d" in decoded   # day 1 (accent1)
    assert "0xa8453e" in decoded   # day 2 (terracotta)
    # Per-day labels (1, 2), not per-stop (1, 2, 3).
    assert "label:1" in decoded
    assert "label:2" in decoded
    assert "label:3" not in decoded


def test_empty_day_drops_marker_but_does_not_shift_day_color():
    # Day 1 empty → no marker emitted, no color slot consumed; day 2 stays terracotta.
    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[
            [],
            [(41.0, 20.0)],
        ],
        polylines=[],
    )
    decoded = unquote(url)
    assert decoded.count("markers=") == 1
    # The emitted marker is at day-index 1 → terracotta and label 2.
    assert "0xa8453e" in decoded
    assert "label:2" in decoded


def test_label_for_anchor_alphabet_rule():
    # 1..9 for the first nine days.
    assert [_label_for_anchor(i) for i in range(9)] == list("123456789")
    # A..Z for the next 26 days (theoretical; real trips are <= a handful).
    assert [_label_for_anchor(9 + i) for i in range(26)] == list(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    )
    # Unlabeled past day 35 — the visual is fine without it.
    assert _label_for_anchor(35) is None
    assert _label_for_anchor(100) is None


def test_url_includes_app_palette_style_params():
    url = build_static_map_url(
        api_key="k",
        signing_secret="",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[],
    )
    decoded = unquote(url)
    # AppColors-derived base-map style applied: landscape uses AppColors.offWhite,
    # POI dots are off so they don't fight our day anchors.
    assert "feature:landscape|color:0xF8ECE5" in decoded
    assert "feature:poi|visibility:off" in decoded
    assert "feature:water|color:0x9BB7C2" in decoded


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
