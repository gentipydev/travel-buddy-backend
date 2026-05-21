"""Tests for `app.domain.trips.mutations`.

8 mutator functions cover whole-trip replace + 7 surgical patches that the
chat refine path applies one-per-tool-use-block. The repo helpers
(`_load_trip`, `_load_day`, `_load_stop`, `_touch_trip`) are exercised
indirectly through FakeSession — we queue the rows each SELECT should
return and verify the mutator's flush/add/delete signature.

Coverage targets per docs/07 item 5: happy path + 1-2 edge cases per
mutator (empty trip, out-of-bounds index, last-stop deletion, auth fail,
duplicate day_number, etc.).
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from fastapi import HTTPException

from app.db.models import Trip, TripDay, TripStop
from app.domain.trips.mutations import (
    apply_add_day,
    apply_delete_day,
    apply_delete_stop,
    apply_insert_stop,
    apply_reorder_stops,
    apply_update_stop,
    apply_update_trip,
    replace_trip_from_payload,
)

# conftest.py:31-34 auto-marks async tests but the hook isn't firing under
# pytest-8 + anyio-4.6; module-level mark is the reliable fallback.
pytestmark = pytest.mark.anyio


_USER_ID = 7
_OTHER_USER_ID = 99


def _make_stop(
    *, name: str = "Berat Castle", kind: str = "sight", sort_order: int = 0,
    stop_id: uuid.UUID | None = None,
) -> TripStop:
    """Build a TripStop with sensible defaults. id is preassigned so tests
    can address it from the outside (SQLAlchemy normally fills id on flush,
    but FakeSession never flushes)."""
    return TripStop(
        id=stop_id or uuid.uuid4(),
        sort_order=sort_order,
        name=name,
        kind=kind,
    )


def _make_day(
    *, day_number: int = 1, stops: list[TripStop] | None = None,
    day_id: uuid.UUID | None = None,
) -> TripDay:
    day = TripDay(
        id=day_id or uuid.uuid4(),
        day_number=day_number,
        title=None,
        base_city=None,
    )
    # Bypass the SA relationship machinery — assigning a Python list works for
    # FakeSession's "no real ORM" world.
    day.stops = list(stops) if stops else []
    return day


def _make_trip(*, days: list[TripDay] | None = None) -> Trip:
    now = datetime.now(UTC)
    trip = Trip(
        id=uuid.uuid4(),
        user_id=_USER_ID,
        title="Test trip",
        summary=None,
        start_date=None,
        created_at=now,
        updated_at=now,
    )
    trip.days = list(days) if days else []
    return trip


# ---------------------------------------------------------------------------
# replace_trip_from_payload
# ---------------------------------------------------------------------------


async def test_replace_trip_from_payload_overwrites_top_fields_and_days(fake_db):
    trip = _make_trip(days=[_make_day(stops=[_make_stop(name="old stop")])])
    refreshed = _make_trip()  # post-replace reload
    fake_db.queue(trip)
    fake_db.queue(refreshed)
    payload = {
        "title": "Brand new title",
        "summary": "A summary",
        "start_date": "2026-06-01",
        "days": [
            {
                "day_number": 1,
                "title": "Day 1",
                "stops": [
                    {"name": "Berat Castle", "kind": "sight"},
                    {"name": "Onufri Museum", "kind": "sight"},
                ],
            }
        ],
    }
    result = await replace_trip_from_payload(
        fake_db, trip_id=trip.id, user_id=_USER_ID, payload=payload
    )
    assert result is refreshed
    # Top-level fields written before the children were rebuilt.
    assert trip.title == "Brand new title"
    assert trip.summary == "A summary"
    assert trip.start_date.isoformat() == "2026-06-01"
    # Children swapped out — `trip.days = []` then append the new ones.
    assert len(trip.days) == 1
    assert trip.days[0].day_number == 1
    assert [s.name for s in trip.days[0].stops] == [
        "Berat Castle", "Onufri Museum"
    ]
    # Per-stop sort_order is auto-assigned from the enumeration index.
    assert [s.sort_order for s in trip.days[0].stops] == [0, 1]


async def test_replace_trip_404_when_trip_not_owned(fake_db):
    """Auth gate: _load_trip returns None (the join filters out wrong owner)
    → 404, not 403, so a guessed UUID reveals nothing about existence."""
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await replace_trip_from_payload(
            fake_db, trip_id=uuid.uuid4(), user_id=_OTHER_USER_ID,
            payload={"title": "x", "days": []},
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# apply_update_trip
# ---------------------------------------------------------------------------


async def test_apply_update_trip_writes_only_set_fields(fake_db):
    trip = _make_trip()
    original_summary = trip.summary
    fake_db.queue(trip)
    fake_db.queue(trip)
    await apply_update_trip(
        fake_db, trip_id=trip.id, user_id=_USER_ID,
        payload={"title": "Renamed"},  # exclude_unset=True at dispatcher
    )
    assert trip.title == "Renamed"
    # summary not in payload → untouched.
    assert trip.summary is original_summary


async def test_apply_update_trip_404_when_missing(fake_db):
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await apply_update_trip(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID,
            payload={"title": "x"},
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# apply_add_day
# ---------------------------------------------------------------------------


async def test_apply_add_day_inserts_new_day_with_stops(fake_db):
    trip = _make_trip(days=[_make_day(day_number=1)])
    refreshed = _make_trip()
    fake_db.queue(trip)
    fake_db.queue(refreshed)
    await apply_add_day(
        fake_db, trip_id=trip.id, user_id=_USER_ID,
        payload={
            "day_number": 2,
            "title": "Drive to Berat",
            "stops": [{"name": "Lunch", "kind": "food"}],
        },
    )
    # The new day is the only object passed to db.add — its stops travel
    # along via the relationship rather than being added individually.
    added_days = [obj for obj in fake_db.added if isinstance(obj, TripDay)]
    assert len(added_days) == 1
    new_day = added_days[0]
    assert new_day.day_number == 2
    assert new_day.title == "Drive to Berat"
    assert len(new_day.stops) == 1
    assert new_day.stops[0].name == "Lunch"
    assert new_day.stops[0].sort_order == 0


async def test_apply_add_day_rejects_duplicate_day_number(fake_db):
    """day_number is UNIQUE per trip — the mutator pre-checks rather than
    letting the SA UNIQUE constraint raise an opaque IntegrityError."""
    trip = _make_trip(days=[_make_day(day_number=1)])
    fake_db.queue(trip)
    with pytest.raises(HTTPException) as exc:
        await apply_add_day(
            fake_db, trip_id=trip.id, user_id=_USER_ID,
            payload={"day_number": 1},  # collides
        )
    assert exc.value.status_code == 409
    assert "day_number already exists" in exc.value.detail


# ---------------------------------------------------------------------------
# apply_delete_day
# ---------------------------------------------------------------------------


async def test_apply_delete_day_deletes_target_day(fake_db):
    day = _make_day(day_number=1)
    refreshed = _make_trip()
    fake_db.queue(day)         # _load_day
    fake_db.queue(refreshed)   # _load_trip refresh
    await apply_delete_day(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, day_id=day.id,
    )
    assert day in fake_db.deleted


async def test_apply_delete_day_404_when_day_missing(fake_db):
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await apply_delete_day(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID,
            day_id=uuid.uuid4(),
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# apply_insert_stop
# ---------------------------------------------------------------------------


async def test_apply_insert_stop_appends_at_end_when_no_sort_order(fake_db):
    """No sort_order → append. Mutator computes max(existing) + 1."""
    day = _make_day(stops=[_make_stop(sort_order=0), _make_stop(sort_order=1)])
    refreshed = _make_trip()
    fake_db.queue(day)
    fake_db.queue(refreshed)
    await apply_insert_stop(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, day_id=day.id,
        payload={"name": "Sunset bar", "kind": "food"},
    )
    added_stops = [o for o in fake_db.added if isinstance(o, TripStop)]
    assert len(added_stops) == 1
    assert added_stops[0].sort_order == 2  # max(0, 1) + 1


async def test_apply_insert_stop_with_sort_order_shifts_existing(fake_db):
    """Explicit sort_order=1 should issue a SHIFT UPDATE on stops >= 1, then
    insert at sort_order=1. We verify the executed statement count + the
    inserted stop's sort_order."""
    day = _make_day(stops=[_make_stop(sort_order=0), _make_stop(sort_order=1)])
    refreshed = _make_trip()
    fake_db.queue(day)
    fake_db.queue(refreshed)
    await apply_insert_stop(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, day_id=day.id,
        payload={"name": "Midday cafe", "kind": "food", "sort_order": 1},
    )
    added_stops = [o for o in fake_db.added if isinstance(o, TripStop)]
    assert added_stops[0].sort_order == 1
    # 4 executes: _load_day (SELECT) + shift UPDATE + _touch_trip UPDATE +
    # _load_trip (SELECT). DML doesn't pop the queue so the 2 SELECTs map to
    # the 2 queued items.
    assert len(fake_db.executed) == 4


async def test_apply_insert_stop_404_when_day_missing(fake_db):
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await apply_insert_stop(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID,
            day_id=uuid.uuid4(), payload={"name": "x", "kind": "sight"},
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# apply_update_stop
# ---------------------------------------------------------------------------


async def test_apply_update_stop_patches_set_keys_only(fake_db):
    """Untouched fields (city, lat, lng, google_place_id) stay as-is —
    the dispatcher passes exclude_unset=True so absent keys are absent."""
    stop = _make_stop(name="Old name", kind="sight")
    stop.city = "Berat"
    refreshed = _make_trip()
    fake_db.queue(stop)
    fake_db.queue(refreshed)
    await apply_update_stop(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, stop_id=stop.id,
        payload={"name": "New name"},  # name only
    )
    assert stop.name == "New name"
    assert stop.city == "Berat"  # untouched


async def test_apply_update_stop_404_when_stop_missing(fake_db):
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await apply_update_stop(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID,
            stop_id=uuid.uuid4(), payload={"name": "x"},
        )
    assert exc.value.status_code == 404


async def test_apply_update_stop_sets_completed_at(fake_db):
    """Mobile mark-visited path: PATCH carries `completed_at` (UTC datetime);
    setattr loop writes it without touching other fields."""
    stop = _make_stop(name="Skanderbeg Square", kind="sight")
    stop.city = "Tirana"
    stop.completed_at = None
    refreshed = _make_trip()
    fake_db.queue(stop)
    fake_db.queue(refreshed)
    visited_at = datetime.now(UTC)
    await apply_update_stop(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, stop_id=stop.id,
        payload={"completed_at": visited_at},
    )
    assert stop.completed_at == visited_at
    assert stop.name == "Skanderbeg Square"  # untouched
    assert stop.city == "Tirana"  # untouched


async def test_apply_update_stop_resets_completed_at_to_null(fake_db):
    """Manual override "Reset to upcoming": PATCH sends completed_at=None;
    setattr writes NULL. Other fields untouched."""
    stop = _make_stop(name="Et'hem Bey Mosque", kind="sight")
    stop.completed_at = datetime.now(UTC)
    refreshed = _make_trip()
    fake_db.queue(stop)
    fake_db.queue(refreshed)
    await apply_update_stop(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, stop_id=stop.id,
        payload={"completed_at": None},
    )
    assert stop.completed_at is None
    assert stop.name == "Et'hem Bey Mosque"  # untouched


# ---------------------------------------------------------------------------
# apply_delete_stop
# ---------------------------------------------------------------------------


async def test_apply_delete_stop_deletes_target_and_leaves_gaps(fake_db):
    """Per design: do NOT compact sort_orders on delete. The relationship's
    order_by="TripStop.sort_order" handles the gap. Verify no UPDATE was
    issued (only _load_stop SELECT + _touch_trip UPDATE + _load_trip SELECT)."""
    stop = _make_stop(sort_order=1)
    refreshed = _make_trip()
    fake_db.queue(stop)
    fake_db.queue(refreshed)
    await apply_delete_stop(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, stop_id=stop.id,
    )
    assert stop in fake_db.deleted
    # 3 executes: _load_stop (SELECT) + _touch_trip (UPDATE) + _load_trip (SELECT).
    # No compaction UPDATE means dashboards can address by stop_id forever.
    assert len(fake_db.executed) == 3


async def test_apply_delete_stop_404_when_stop_missing(fake_db):
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await apply_delete_stop(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID,
            stop_id=uuid.uuid4(),
        )
    assert exc.value.status_code == 404


# ---------------------------------------------------------------------------
# apply_reorder_stops
# ---------------------------------------------------------------------------


async def test_apply_reorder_stops_accepts_permutation(fake_db):
    """A valid permutation (0..N-1, same set of ids) triggers the two-phase
    SHIFT + CASE update. We assert the two UPDATEs were issued."""
    s1, s2 = _make_stop(sort_order=0), _make_stop(sort_order=1)
    day = _make_day(stops=[s1, s2])
    refreshed = _make_trip()
    fake_db.queue(day)
    fake_db.queue(refreshed)
    await apply_reorder_stops(
        fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, day_id=day.id,
        payload={"order": [
            {"stop_id": str(s2.id), "sort_order": 0},
            {"stop_id": str(s1.id), "sort_order": 1},
        ]},
    )
    # 5 executes: _load_day (SELECT) + SHIFT UPDATE + CASE UPDATE +
    # _touch_trip UPDATE + _load_trip (SELECT).
    assert len(fake_db.executed) == 5


async def test_apply_reorder_stops_rejects_unknown_stop_id(fake_db):
    """The body must address EXACTLY the stops on the day — set mismatch
    means a concurrent insert/delete and we refuse rather than half-applying."""
    s1 = _make_stop(sort_order=0)
    day = _make_day(stops=[s1])
    fake_db.queue(day)
    with pytest.raises(HTTPException) as exc:
        await apply_reorder_stops(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, day_id=day.id,
            payload={"order": [
                {"stop_id": str(uuid.uuid4()), "sort_order": 0},  # not on day
            ]},
        )
    assert exc.value.status_code == 409
    assert "stop set out of sync" in exc.value.detail


async def test_apply_reorder_stops_rejects_non_contiguous_orders(fake_db):
    """sort_order must be 0..N-1 contiguous — gaps or duplicates are rejected
    so the post-condition is always a clean dense numbering."""
    s1, s2 = _make_stop(sort_order=0), _make_stop(sort_order=1)
    day = _make_day(stops=[s1, s2])
    fake_db.queue(day)
    with pytest.raises(HTTPException) as exc:
        await apply_reorder_stops(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID, day_id=day.id,
            payload={"order": [
                {"stop_id": str(s1.id), "sort_order": 0},
                {"stop_id": str(s2.id), "sort_order": 5},  # gap
            ]},
        )
    assert exc.value.status_code == 400
    assert "contiguous" in exc.value.detail


async def test_apply_reorder_stops_404_when_day_missing(fake_db):
    fake_db.queue(None)
    with pytest.raises(HTTPException) as exc:
        await apply_reorder_stops(
            fake_db, trip_id=uuid.uuid4(), user_id=_USER_ID,
            day_id=uuid.uuid4(), payload={"order": []},
        )
    assert exc.value.status_code == 404
