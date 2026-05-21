"""Tests for the free-tier `MAX_TRIPS_PER_USER` cap on `POST /trips`.

The cap is enforced server-side as the source of truth — a forked or stale
mobile client without the matching pre-flight check still bounces off the
same 409 + `trip_limit_reached` detail token here. The mobile error mapper
keys on that exact substring, so changing it is a breaking contract change.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.auth.supabase_auth import get_current_user
from app.db.engine import get_session
from app.db.models import Trip, User
from app.deps import require_local_user
from app.main import app
from app.routers import trips as trips_module

# conftest's auto-anyio hook is unreliable under anyio 4.6 — the TestClient
# paths in this file are sync anyway, but keep the marker on for parity with
# the rest of the suite.
pytestmark = pytest.mark.anyio


_USER = User(
    id=42,
    supabase_uid="supa-uid",
    email="t@example.com",
    created_at=datetime.now(UTC),
)


@pytest.fixture
def client(fake_db):
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


def _payload() -> dict[str, Any]:
    """Minimal valid TripCreate body. `days` requires min_length=1 and each
    day requires at least one stop with `name` + `kind` (per app.schemas)."""
    return {
        "title": "Test trip",
        "days": [
            {
                "day_number": 1,
                "stops": [{"name": "Test stop", "kind": "sight"}],
            }
        ],
    }


def _patch_post_commit(monkeypatch) -> None:
    """Bypass the post-commit `_load_trip` + `serialize_trip` machinery so
    the happy path doesn't require populating every TripOut field on an ORM
    object that never hit a real DB."""

    async def _fake_load_trip(db, trip_id, user_id) -> Trip:
        now = datetime.now(UTC)
        trip = Trip(
            id=trip_id if isinstance(trip_id, uuid.UUID) else uuid.uuid4(),
            user_id=user_id,
            title="Test trip",
            summary=None,
            start_date=None,
            created_at=now,
            updated_at=now,
        )
        trip.days = []
        return trip

    async def _fake_serialize(db, trip) -> dict[str, Any]:
        return {
            "id": str(trip.id),
            "title": trip.title,
            "days": [],
            "created_at": trip.created_at.isoformat(),
            "updated_at": trip.updated_at.isoformat(),
        }

    monkeypatch.setattr(trips_module, "_load_trip", _fake_load_trip)
    monkeypatch.setattr(trips_module, "serialize_trip", _fake_serialize)


def test_create_trip_rejected_when_user_at_cap(monkeypatch, client, fake_db):
    """User already has 3 trips → POST /trips returns 409 with the stable
    `trip_limit_reached` detail token the mobile mapper keys on."""
    _patch_post_commit(monkeypatch)
    fake_db.queue(3)  # COUNT(Trip.id) for this user returns 3

    response = client.post("/trips", json=_payload())

    assert response.status_code == 409
    assert response.json()["detail"] == "trip_limit_reached"
    # No new trip was added — the handler short-circuited before db.add().
    assert fake_db.added == []


def test_create_trip_allowed_when_under_cap(monkeypatch, client, fake_db):
    """User has 2 trips → POST /trips for a 3rd is accepted (201)."""
    _patch_post_commit(monkeypatch)
    fake_db.queue(2)

    response = client.post("/trips", json=_payload())

    assert response.status_code == 201
    body = response.json()
    assert body["title"] == "Test trip"
    # One Trip ORM was staged for insert.
    assert len(fake_db.added) == 1


def test_create_trip_allowed_when_first_trip(monkeypatch, client, fake_db):
    """User has 0 trips → POST /trips for the first one is accepted. Edge
    case: the `or 0` fallback handles a None scalar (no rows yet)."""
    _patch_post_commit(monkeypatch)
    fake_db.queue(None)  # FakeSession.scalar returns None when count=0 row

    response = client.post("/trips", json=_payload())

    assert response.status_code == 201
