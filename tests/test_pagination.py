"""Tests for keyset pagination on `GET /trips` and `GET /conversations`.

Both endpoints share the same `(updated_at DESC, id DESC)` ordering and the
same opaque cursor format (`"{updated_at_iso}|{uuid}"`). Coverage targets
the contract surface — cursor encoding/decoding, limit clamping, last-page
None terminator — not the underlying ORM (FakeSession returns what we
queue, no real query runs).

The integration paths are tested via FastAPI TestClient with
`require_local_user` + `get_session` overridden. `_serialize` is also
monkey-patched in the trips router so we don't have to populate every
required TripOut field on the test ORM objects.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.auth.supabase_auth import get_current_user
from app.db.engine import get_session
from app.db.models import Conversation, Trip, User
from app.deps import require_local_user
from app.main import app
from app.routers import trips as trips_module

# This file mixes sync TestClient calls (no anyio needed) with async fixtures
# elsewhere; both work without a module-level marker.


_USER = User(
    id=42,
    supabase_uid="supa-uid",
    email="t@example.com",
    created_at=datetime.now(UTC),
)


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def client(fake_db):
    """TestClient with require_local_user + get_session overridden.

    `get_current_user` is also overridden so the supabase JWT verification
    doesn't run (it would 401 with no Bearer token); `require_local_user`
    short-circuits to our fixed User."""

    def _user_dep():
        return _USER

    def _supa_dep():
        return {"id": _USER.supabase_uid}

    def _db_dep():
        return fake_db

    app.dependency_overrides[require_local_user] = _user_dep
    app.dependency_overrides[get_current_user] = _supa_dep
    app.dependency_overrides[get_session] = _db_dep
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        app.dependency_overrides.pop(require_local_user, None)
        app.dependency_overrides.pop(get_current_user, None)
        app.dependency_overrides.pop(get_session, None)


def _make_trip(*, updated_at: datetime, trip_id: uuid.UUID | None = None) -> Trip:
    now = datetime.now(UTC)
    trip = Trip(
        id=trip_id or uuid.uuid4(),
        user_id=_USER.id,
        title="trip",
        summary=None,
        start_date=None,
        created_at=now,
        updated_at=updated_at,
    )
    trip.days = []
    return trip


def _make_conv(*, updated_at: datetime, conv_id: uuid.UUID | None = None) -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=conv_id or uuid.uuid4(),
        user_id=_USER.id,
        title=None,
        created_at=now,
        updated_at=updated_at,
    )


def _patch_trip_serialize(monkeypatch) -> None:
    """Skip the real `_serialize` so test trips don't need every TripOut
    field populated. The mock returns the minimal valid TripOut shape."""

    def _fake_serialize(trip: Trip) -> dict[str, Any]:
        return {
            "id": str(trip.id),
            "title": trip.title,
            "days": [],
            "created_at": trip.created_at.isoformat(),
            "updated_at": trip.updated_at.isoformat(),
        }

    monkeypatch.setattr(trips_module, "_serialize", _fake_serialize)


# ---------------------------------------------------------------------------
# GET /trips — happy paths.
# ---------------------------------------------------------------------------


def test_trips_first_page_returns_envelope_with_items(
    monkeypatch, client, fake_db
):
    """No cursor + 3 rows + default limit=20 → all 3 items returned with
    next_cursor=None (since len(rows) ≤ limit)."""
    _patch_trip_serialize(monkeypatch)
    now = datetime.now(UTC)
    fake_db.queue([
        _make_trip(updated_at=now - timedelta(minutes=i))
        for i in range(3)
    ])
    response = client.get("/trips")
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["items"], list)
    assert len(body["items"]) == 3
    assert body["next_cursor"] is None


def test_trips_next_cursor_emitted_when_more_rows_exist(
    monkeypatch, client, fake_db
):
    """limit=2, but we queue 3 (limit+1) → next_cursor is set to the
    second item's updated_at|id so the next page starts AT that boundary."""
    _patch_trip_serialize(monkeypatch)
    now = datetime.now(UTC)
    trips = [
        _make_trip(updated_at=now - timedelta(minutes=i))
        for i in range(3)
    ]
    fake_db.queue(trips)
    response = client.get("/trips?limit=2")
    body = response.json()
    assert len(body["items"]) == 2
    # next_cursor encodes the 2nd row's (updated_at, id).
    expected_ts = trips[1].updated_at.isoformat()
    expected_id = str(trips[1].id)
    assert body["next_cursor"] == f"{expected_ts}|{expected_id}"


def test_trips_cursor_round_trips(monkeypatch, client, fake_db):
    """Send a valid cursor → server parses and returns 200. (FakeSession
    doesn't honor the WHERE clause; we're only validating cursor parsing
    + the second-page response shape.)"""
    _patch_trip_serialize(monkeypatch)
    fake_db.queue([])  # empty second page
    cursor = f"{datetime.now(UTC).isoformat()}|{uuid.uuid4()}"
    response = client.get("/trips", params={"cursor": cursor})
    assert response.status_code == 200
    body = response.json()
    assert body["items"] == []
    assert body["next_cursor"] is None


# ---------------------------------------------------------------------------
# GET /trips — error paths.
# ---------------------------------------------------------------------------


def test_trips_invalid_cursor_returns_400(monkeypatch, client, fake_db):
    """Cursor with no `|` separator → _parse_cursor raises HTTPException(400).
    The error body is JSON with `detail`."""
    _patch_trip_serialize(monkeypatch)
    response = client.get("/trips", params={"cursor": "not-a-valid-cursor"})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid cursor format"


def test_trips_invalid_cursor_uuid_returns_400(monkeypatch, client, fake_db):
    """Cursor with valid `|` split but a non-UUID id → 400."""
    _patch_trip_serialize(monkeypatch)
    bad = f"{datetime.now(UTC).isoformat()}|not-a-uuid"
    response = client.get("/trips", params={"cursor": bad})
    assert response.status_code == 400


def test_trips_limit_too_large_returns_422(monkeypatch, client, fake_db):
    """Pydantic Query(le=100) → 422 for limit=101. No DB call attempted."""
    response = client.get("/trips?limit=101")
    assert response.status_code == 422


def test_trips_limit_zero_returns_422(monkeypatch, client, fake_db):
    """Pydantic Query(ge=1) → 422 for limit=0."""
    response = client.get("/trips?limit=0")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /conversations — same contract.
# ---------------------------------------------------------------------------


def test_conversations_first_page_returns_envelope(client, fake_db):
    now = datetime.now(UTC)
    fake_db.queue([_make_conv(updated_at=now - timedelta(minutes=i)) for i in range(2)])
    response = client.get("/conversations")
    assert response.status_code == 200
    body = response.json()
    assert len(body["items"]) == 2
    assert body["next_cursor"] is None
    # Item shape matches the documented dict, not a full Conversation model.
    item = body["items"][0]
    assert set(item.keys()) >= {
        "id", "title", "trip_id", "created_at", "updated_at", "message_count",
    }


def test_conversations_next_cursor_when_more_rows(client, fake_db):
    now = datetime.now(UTC)
    convs = [
        _make_conv(updated_at=now - timedelta(minutes=i))
        for i in range(3)
    ]
    fake_db.queue(convs)
    response = client.get("/conversations?limit=2")
    body = response.json()
    assert len(body["items"]) == 2
    expected_ts = convs[1].updated_at.isoformat()
    expected_id = str(convs[1].id)
    assert body["next_cursor"] == f"{expected_ts}|{expected_id}"


def test_conversations_invalid_cursor_returns_400(client, fake_db):
    response = client.get("/conversations", params={"cursor": "garbage"})
    assert response.status_code == 400


def test_conversations_limit_too_large_returns_422(client, fake_db):
    response = client.get("/conversations?limit=101")
    assert response.status_code == 422


# ---------------------------------------------------------------------------
# Cursor format roundtrip — sanity check on the encoding shared by both routes.
# ---------------------------------------------------------------------------


def test_parse_cursor_roundtrips_iso_and_uuid():
    """The format `"{updated_at_iso}|{uuid}"` is opaque to the client; we
    own both encoding (in the route bodies) and decoding (`_parse_cursor`).
    Verify they're inverse so a `next_cursor` echoed back as `?cursor=`
    always parses cleanly."""
    ts = datetime.now(UTC)
    row_id = uuid.uuid4()
    encoded = f"{ts.isoformat()}|{row_id}"
    parsed_ts, parsed_id = trips_module._parse_cursor(encoded)
    assert parsed_ts == ts
    assert parsed_id == row_id


def test_parse_cursor_rejects_empty_string():
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc:
        trips_module._parse_cursor("")
    assert exc.value.status_code == 400
