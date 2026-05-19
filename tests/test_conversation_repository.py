"""Tests for `get_or_create_conversation`'s per-user 3-cap eviction.

Uses the in-repo `FakeSession` fixture (`tests/conftest.py`) to queue SELECT
results and inspect the statements the production code issued. The cap is
on `trip_id IS NULL` chats only — trip-linked refine threads are exempt
from both counting and eviction. Test #6 is a regression guard for that
contract.
"""
from __future__ import annotations

import uuid

import pytest
from sqlalchemy.sql.dml import Delete
from sqlalchemy.sql.expression import Select

from app.db.models import Conversation
from app.domain.conversations.repository import (
    _MAX_CONVERSATIONS_PER_USER,
    get_or_create_conversation,
)

pytestmark = pytest.mark.anyio

_USER_ID = 42


def _selects(fake_db) -> list:
    return [s for s in fake_db.executed if isinstance(s, Select)]


def _deletes(fake_db) -> list:
    return [s for s in fake_db.executed if isinstance(s, Delete)]


async def test_first_conversation_creates_without_eviction(fake_db):
    fake_db.queue(0)  # count → to_evict = -2 → skipped

    conv = await get_or_create_conversation(fake_db, None, _USER_ID)

    assert isinstance(conv, Conversation)
    assert conv.user_id == _USER_ID
    assert _deletes(fake_db) == []
    assert len(fake_db.added) == 1
    assert fake_db.flushed == 1


async def test_below_cap_creates_without_eviction(fake_db):
    fake_db.queue(2)  # count → to_evict = 0 → skipped

    await get_or_create_conversation(fake_db, None, _USER_ID)

    assert _deletes(fake_db) == []
    assert len(fake_db.added) == 1
    assert fake_db.flushed == 1


async def test_at_cap_evicts_one_oldest(fake_db):
    oldest_id = uuid.uuid4()
    fake_db.queue(3)               # count
    fake_db.queue([oldest_id])     # oldest_ids → 1 eviction

    await get_or_create_conversation(fake_db, None, _USER_ID)

    deletes = _deletes(fake_db)
    assert len(deletes) == 1
    compiled = str(deletes[0].compile(compile_kwargs={"literal_binds": True}))
    # literal_binds renders UUIDs as bare hex (no dashes), e.g. 'abc...def'.
    assert oldest_id.hex in compiled
    assert len(fake_db.added) == 1


async def test_legacy_user_above_cap_trims_to_three(fake_db):
    # User had 7 chats → must drop 5 in one DELETE so post-insert count is 3.
    oldest_ids = [uuid.uuid4() for _ in range(5)]
    fake_db.queue(7)
    fake_db.queue(oldest_ids)

    await get_or_create_conversation(fake_db, None, _USER_ID)

    deletes = _deletes(fake_db)
    assert len(deletes) == 1, "expect a single bulk DELETE, not a loop"
    compiled = str(deletes[0].compile(compile_kwargs={"literal_binds": True}))
    for oid in oldest_ids:
        assert oid.hex in compiled, f"missing id {oid} in DELETE: {compiled}"
    assert len(fake_db.added) == 1


async def test_existing_conversation_id_returns_without_eviction(fake_db):
    existing_id = uuid.uuid4()
    existing = Conversation(id=existing_id, user_id=_USER_ID, messages=[])
    fake_db.queue(existing)  # initial conv lookup hits

    conv = await get_or_create_conversation(fake_db, existing_id, _USER_ID)

    assert conv is existing
    assert _deletes(fake_db) == []
    assert fake_db.added == []
    assert fake_db.flushed == 0


async def test_count_and_eviction_filter_trip_id_is_null(fake_db):
    # Regression guard: both SELECTs in the eviction path must filter on
    # trip_id IS NULL so trip-linked refine chats are exempt from the cap.
    fake_db.queue(3)
    fake_db.queue([uuid.uuid4()])

    await get_or_create_conversation(fake_db, None, _USER_ID)

    selects = _selects(fake_db)
    assert len(selects) == 2, "expect count SELECT + oldest_ids SELECT"
    for stmt in selects:
        sql = str(stmt.compile(compile_kwargs={"literal_binds": True}))
        assert "trip_id IS NULL" in sql, f"missing trip exemption in: {sql}"


def test_cap_constant_value():
    """Lock the cap value so a future bump can't sneak past code review."""
    assert _MAX_CONVERSATIONS_PER_USER == 3
