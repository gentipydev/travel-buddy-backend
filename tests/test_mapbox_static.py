"""Unit tests for app.integrations.mapbox_static.

Pure URL construction — no DB, no HTTP. The downsampler and the overlay
syntax are the load-bearing details; the markers/polylines paths are
exercised incidentally so the regression net is wide.
"""
from urllib.parse import unquote

from app.integrations.mapbox_static import (
    _label_for_anchor,
    build_static_map_url,
    decode_polyline,
    encode_polyline,
)


def test_url_emits_one_pin_per_day_at_centroid():
    # Day 1: two stops, centroid (40.55, 19.55). Day 2: one stop (its own pos).
    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[
            [(40.5, 19.5), (40.6, 19.6)],
            [(41.0, 20.0)],
        ],
        polylines=[],
    )
    decoded = unquote(url)
    # Two non-empty days → exactly two pins (down from per-stop = 3).
    assert decoded.count("pin-s-") == 2
    # Mapbox takes the anchor longitude-first.
    assert "(19.55000,40.55000)" in decoded
    assert "(20.00000,41.00000)" in decoded
    # Per-day colors preserved, bare hex (no 0x / # prefix).
    assert "+e4935d" in decoded   # day 1 (accent1)
    assert "+a8453e" in decoded   # day 2 (terracotta)
    # Per-day labels (1, 2), not per-stop (1, 2, 3).
    assert "pin-s-1+" in decoded
    assert "pin-s-2+" in decoded
    assert "pin-s-3+" not in decoded


def test_empty_day_drops_pin_but_does_not_shift_day_color():
    # Day 1 empty → no pin emitted, no color slot consumed; day 2 stays terracotta.
    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[
            [],
            [(41.0, 20.0)],
        ],
        polylines=[],
    )
    decoded = unquote(url)
    assert decoded.count("pin-s-") == 1
    # The emitted pin is at day-index 1 → terracotta and label 2.
    assert "pin-s-2+a8453e" in decoded


def test_label_for_anchor_alphabet_rule():
    # 1..9 for the first nine days.
    assert [_label_for_anchor(i) for i in range(9)] == list("123456789")
    # a..z for the next 26 days (theoretical; real trips are <= a handful).
    # Lowercase because that is what `pin-s-{label}` accepts.
    assert [_label_for_anchor(9 + i) for i in range(26)] == list(
        "abcdefghijklmnopqrstuvwxyz"
    )
    # Unlabeled past day 35 — the visual is fine without it.
    assert _label_for_anchor(35) is None
    assert _label_for_anchor(100) is None


def test_url_targets_a_static_images_style_with_auto_viewport():
    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[],
    )
    assert url.startswith("https://api.mapbox.com/styles/v1/mapbox/light-v11/static/")
    # `auto` fits every overlay — no centre/zoom math.
    assert "/auto/600x300?" in url
    assert url.endswith("?access_token=pk.test")


def test_url_includes_polylines_with_distinct_colors():
    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=["abc", "def"],
    )
    decoded = unquote(url)
    assert decoded.count("path-4+") == 2
    assert "(abc)" in decoded
    assert "(def)" in decoded
    # Distinct day colors, in the order the caller passed them.
    assert "path-4+e4935d-0.9(abc)" in decoded
    assert "path-4+a8453e-0.9(def)" in decoded


def test_polyline_is_percent_encoded_in_the_overlay():
    # The polyline alphabet includes characters that would otherwise terminate
    # or corrupt the URL — `?` starts the query, `\` and `|` are unsafe.
    enc = encode_polyline([(40.0, 19.0), (41.5, 20.5)])
    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[],
        polylines=[enc],
    )
    # Nothing raw leaks through: the only `?` is the one before access_token.
    assert url.count("?") == 1
    assert "\\" not in url
    # ...but it round-trips back to the exact polyline we passed in.
    overlay = unquote(url).split("static/")[1].split("/auto/")[0]
    assert overlay == f"path-4+e4935d-0.9({enc})"


def test_url_downsamples_when_too_long():
    # Construct a single very long polyline. The encoder produces ~5-7 bytes
    # per point, so 2000 points across [40, 41]×[19, 20] gives a polyline
    # that overshoots the 7800-char URL cap by a comfortable margin.
    points = [(40.0 + i * 0.0005, 19.0 + i * 0.0005) for i in range(2000)]
    enc = encode_polyline(points)
    assert len(enc) > 7900   # sanity: input is genuinely oversized

    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[[(40.5, 19.5)]],
        polylines=[enc],
    )
    assert len(url) < 7900   # downsampled or polyline-dropped to fit
    # Pins are never dropped by the downsampler.
    assert "(19.50000,40.50000)" in unquote(url)


def test_no_overlays_still_builds_a_valid_url():
    # A trip with no geo'd stops never reaches here (the caller returns None),
    # but the builder must not emit a stray empty path segment if it does.
    url = build_static_map_url(
        access_token="pk.test",
        markers_by_day=[],
        polylines=[],
    )
    assert url == (
        "https://api.mapbox.com/styles/v1/mapbox/light-v11/static"
        "/auto/600x300?access_token=pk.test"
    )


def test_polyline_encode_decode_round_trips():
    # Sanity for the helper; the downsampler depends on this being correct.
    pts = [(40.5, 19.5), (40.6, 19.7), (40.55, 19.65)]
    enc = encode_polyline(pts)
    decoded = decode_polyline(enc)
    assert len(decoded) == len(pts)
    for (lat1, lng1), (lat2, lng2) in zip(pts, decoded, strict=False):
        assert abs(lat1 - lat2) < 1e-4
        assert abs(lng1 - lng2) < 1e-4
