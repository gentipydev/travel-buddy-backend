"""Unit tests for app.services.itinerary_validator.

Pure pydantic — no FastAPI client, no async, no Anthropic mocks. The validator
is a thin shape-and-schema gate; tests pin the behaviors the SSE generator and
Flutter client both depend on.
"""

from app.services.itinerary_validator import validate_itinerary


def test_valid_minimal_itinerary_passes():
    raw = {
        "title": "Quick weekend",
        "days": [
            {
                "day_number": 1,
                "stops": [{"name": "Berat castle", "kind": "sight"}],
            }
        ],
    }
    result = validate_itinerary(raw)
    assert result.ok
    assert result.error is None
    assert result.payload["title"] == "Quick weekend"
    assert result.payload["days"][0]["stops"][0]["name"] == "Berat castle"


def test_valid_full_itinerary_round_trips():
    raw = {
        "title": "Full Albania loop",
        "summary": "A week through the south",
        "start_date": "2026-06-12",
        "days": [
            {
                "day_number": 1,
                "title": "Tirana → Berat",
                "base_city": "Berat",
                "stops": [
                    {
                        "name": "Berat castle",
                        "kind": "sight",
                        "city": "Berat",
                        "address": "Rruga Mihal Komnena",
                        "duration_minutes": 120,
                        "arrival_time": "09:30",
                        "end_time": "11:30",
                        "transportation": "walk",
                        "notes": "Wear sturdy shoes",
                        "lat": 40.7038,
                        "lng": 19.9457,
                        "google_place_id": "ChIJxx",
                    }
                ],
            }
        ],
    }
    result = validate_itinerary(raw)
    assert result.ok
    # date → ISO string via model_dump(mode="json")
    assert result.payload["start_date"] == "2026-06-12"
    stop = result.payload["days"][0]["stops"][0]
    # Every optional field survives the round trip.
    for key in (
        "city",
        "address",
        "duration_minutes",
        "arrival_time",
        "end_time",
        "transportation",
        "notes",
        "lat",
        "lng",
        "google_place_id",
    ):
        assert key in stop, f"{key} missing from validated payload"


def test_missing_required_field_fails_with_useful_error():
    raw = {
        "title": "x",
        "days": [{"day_number": 1, "stops": [{"kind": "sight"}]}],
    }
    result = validate_itinerary(raw)
    assert not result.ok
    # The offending field path must surface for triage.
    assert "name" in result.error


def test_unknown_kind_fails():
    raw = {
        "title": "x",
        "days": [
            {"day_number": 1, "stops": [{"name": "y", "kind": "rave"}]}
        ],
    }
    assert not validate_itinerary(raw).ok


def test_malformed_arrival_time_fails():
    raw = {
        "title": "x",
        "days": [
            {
                "day_number": 1,
                "stops": [
                    {
                        "name": "y",
                        "kind": "sight",
                        "arrival_time": "morning",
                    }
                ],
            }
        ],
    }
    assert not validate_itinerary(raw).ok


def test_malformed_start_date_fails():
    raw = {
        "title": "x",
        "start_date": "June 12",
        "days": [
            {"day_number": 1, "stops": [{"name": "y", "kind": "sight"}]}
        ],
    }
    assert not validate_itinerary(raw).ok


def test_non_dict_input_does_not_crash():
    for bad in [None, "string", 42, ["array"]]:
        result = validate_itinerary(bad)
        assert not result.ok
        # A human-readable summary, not a stack trace dump.
        assert result.error
        assert "expected object" in result.error


def test_empty_days_array_fails():
    raw = {"title": "x", "days": []}
    assert not validate_itinerary(raw).ok


def test_payload_excludes_none_values():
    raw = {
        "title": "x",
        "days": [
            {"day_number": 1, "stops": [{"name": "y", "kind": "sight"}]}
        ],
    }
    result = validate_itinerary(raw)
    assert result.ok
    # exclude_none keeps the wire payload tight and matches Flutter's
    # tolerant `as String?` parsing — missing keys behave like nulls.
    assert "summary" not in result.payload
    assert "start_date" not in result.payload
    stop = result.payload["days"][0]["stops"][0]
    assert "address" not in stop
    assert "arrival_time" not in stop
