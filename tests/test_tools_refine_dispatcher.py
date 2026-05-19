"""Tests for `app.tools.refine.dispatcher.apply_refine_tool`.

The dispatcher is pure routing: given a tool name string from an Anthropic
`tool_use` block, validate the input against the matching Pydantic body and
hand off to the corresponding `apply_*` mutator. The mutators themselves
are tested in `test_domain_trips_mutations.py` — here we only verify the
routing + validation surface.

Test seam: each mutator import is monkey-patched on the dispatcher module
with an `AsyncMock`, so calls are recorded and assertions can pin the
exact args. No DB / pydantic mocking — Pydantic validation runs for real.
"""
from __future__ import annotations

from unittest.mock import AsyncMock
from uuid import UUID, uuid4

import pytest
from pydantic import ValidationError

from app.tools.refine import dispatcher as dispatcher_module
from app.tools.refine.dispatcher import apply_refine_tool

# conftest.py:31-34 auto-marks async tests but the hook isn't firing under
# pytest-8 + anyio-4.6 (also why test_google_*.py async tests silently skip).
# Module-level pytestmark is the standard fallback.
pytestmark = pytest.mark.anyio


_TRIP_ID = uuid4()
_DAY_ID = uuid4()
_STOP_ID = uuid4()
_USER_ID = 42


def _patch_all_mutators(monkeypatch) -> dict[str, AsyncMock]:
    """Replace every mutator on the dispatcher module with an AsyncMock
    that returns a sentinel `Trip` object. Returns the mock dict so each
    test can assert on the relevant call."""
    sentinel_trip = object()  # dispatcher returns it verbatim
    mocks: dict[str, AsyncMock] = {
        name: AsyncMock(return_value=sentinel_trip)
        for name in (
            "apply_update_stop",
            "apply_insert_stop",
            "apply_delete_stop",
            "apply_reorder_stops",
            "apply_add_day",
            "apply_delete_day",
            "apply_update_trip",
        )
    }
    for name, mock in mocks.items():
        monkeypatch.setattr(dispatcher_module, name, mock)
    return mocks


# ---------------------------------------------------------------------------
# Per-tool routing — each `name` string lands on the matching mutator.
# ---------------------------------------------------------------------------


async def test_update_stop_dispatches_to_apply_update_stop(monkeypatch, fake_db):
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="update_stop",
        raw_input={"stop_id": str(_STOP_ID), "name": "New name"},
    )
    mocks["apply_update_stop"].assert_called_once()
    _, kwargs = mocks["apply_update_stop"].call_args
    assert kwargs["stop_id"] == _STOP_ID
    assert kwargs["trip_id"] == _TRIP_ID
    assert kwargs["user_id"] == _USER_ID
    # Only the fields the model considers "set" — stop_id has been stripped.
    assert kwargs["payload"] == {"name": "New name"}


async def test_insert_stop_dispatches_to_apply_insert_stop(monkeypatch, fake_db):
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="insert_stop",
        raw_input={
            "day_id": str(_DAY_ID),
            "name": "Berat Castle",
            "kind": "sight",
        },
    )
    mocks["apply_insert_stop"].assert_called_once()
    _, kwargs = mocks["apply_insert_stop"].call_args
    assert kwargs["day_id"] == _DAY_ID
    # `model_dump()` (no exclude_unset) — defaults like sort_order land too.
    assert kwargs["payload"]["name"] == "Berat Castle"
    assert kwargs["payload"]["kind"] == "sight"


async def test_delete_stop_dispatches_to_apply_delete_stop(monkeypatch, fake_db):
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="delete_stop",
        raw_input={"stop_id": str(_STOP_ID)},
    )
    mocks["apply_delete_stop"].assert_called_once_with(
        fake_db, trip_id=_TRIP_ID, user_id=_USER_ID, stop_id=_STOP_ID
    )


async def test_reorder_stops_dispatches_to_apply_reorder_stops(
    monkeypatch, fake_db
):
    mocks = _patch_all_mutators(monkeypatch)
    a, b = uuid4(), uuid4()
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="reorder_stops",
        raw_input={
            "day_id": str(_DAY_ID),
            "order": [
                {"stop_id": str(a), "sort_order": 0},
                {"stop_id": str(b), "sort_order": 1},
            ],
        },
    )
    mocks["apply_reorder_stops"].assert_called_once()
    _, kwargs = mocks["apply_reorder_stops"].call_args
    assert kwargs["day_id"] == _DAY_ID
    assert len(kwargs["payload"]["order"]) == 2


async def test_add_day_dispatches_to_apply_add_day(monkeypatch, fake_db):
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="add_day",
        raw_input={"day_number": 2},
    )
    mocks["apply_add_day"].assert_called_once()
    _, kwargs = mocks["apply_add_day"].call_args
    assert kwargs["payload"]["day_number"] == 2


async def test_delete_day_dispatches_to_apply_delete_day(monkeypatch, fake_db):
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="delete_day",
        raw_input={"day_id": str(_DAY_ID)},
    )
    mocks["apply_delete_day"].assert_called_once_with(
        fake_db, trip_id=_TRIP_ID, user_id=_USER_ID, day_id=_DAY_ID
    )


async def test_update_trip_dispatches_to_apply_update_trip(monkeypatch, fake_db):
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="update_trip",
        raw_input={"title": "Renamed"},
    )
    mocks["apply_update_trip"].assert_called_once()
    _, kwargs = mocks["apply_update_trip"].call_args
    # exclude_unset=True means only the set field lands.
    assert kwargs["payload"] == {"title": "Renamed"}


# ---------------------------------------------------------------------------
# Failure modes.
# ---------------------------------------------------------------------------


async def test_unknown_tool_name_raises_value_error(monkeypatch, fake_db):
    """Defense-in-depth: even if a future Anthropic SDK shape change leaks a
    tool name we don't recognize, the dispatcher refuses rather than silently
    proceeding. The chat agent's outer try/except catches and rolls back."""
    _patch_all_mutators(monkeypatch)
    with pytest.raises(ValueError, match="Unknown refine tool name"):
        await apply_refine_tool(
            fake_db,
            trip_id=_TRIP_ID,
            user_id=_USER_ID,
            name="rebuild_universe",
            raw_input={},
        )


async def test_invalid_input_raises_validation_error(monkeypatch, fake_db):
    """Pydantic validation runs for real — missing `name` on insert_stop
    (a required field on TripStopCreateBody) must surface as ValidationError
    so the chat agent emits `event: refine_failed` rather than half-applying."""
    _patch_all_mutators(monkeypatch)
    with pytest.raises(ValidationError):
        await apply_refine_tool(
            fake_db,
            trip_id=_TRIP_ID,
            user_id=_USER_ID,
            name="insert_stop",
            raw_input={"day_id": str(_DAY_ID), "kind": "sight"},
            # `name` is required — missing it must trip Pydantic.
        )


async def test_no_other_mutator_called_when_dispatching(monkeypatch, fake_db):
    """A `delete_stop` call must not touch any other mutator; this catches
    a future refactor that accidentally fell through if/elif branches."""
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="delete_stop",
        raw_input={"stop_id": str(_STOP_ID)},
    )
    for name, mock in mocks.items():
        if name == "apply_delete_stop":
            assert mock.call_count == 1
        else:
            assert mock.call_count == 0, (
                f"{name} was called but should not have been"
            )


async def test_uuid_strings_in_raw_input_are_parsed(monkeypatch, fake_db):
    """Anthropic emits UUIDs as strings inside tool_use.input; the dispatcher
    must coerce them to UUID before passing to mutators (mutators type-hint
    `UUID`)."""
    mocks = _patch_all_mutators(monkeypatch)
    await apply_refine_tool(
        fake_db,
        trip_id=_TRIP_ID,
        user_id=_USER_ID,
        name="delete_day",
        raw_input={"day_id": str(_DAY_ID)},  # string form
    )
    _, kwargs = mocks["apply_delete_day"].call_args
    assert isinstance(kwargs["day_id"], UUID)
    assert kwargs["day_id"] == _DAY_ID
