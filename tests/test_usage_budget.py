"""Unit tests for `app/domain/usage_budget.py`.

Budget gates (per-user daily USD, global ceiling + cache), the fail-soft
`record_usage` upsert, and the per-user concurrent-stream slots.
Uses `FakeSession` from conftest: the gates are SELECTs (queue a scalar),
the upsert is DML (recorded in `.executed`, queue untouched).
"""
import time

import pytest

from app.domain import usage_budget
from app.domain.usage_budget import (
    BudgetExceeded,
    acquire_stream_slot,
    check_global_ceiling,
    check_user_budget,
    record_usage,
    release_stream_slot,
)

pytestmark = pytest.mark.anyio


@pytest.fixture(autouse=True)
def _clean_module_state():
    """Both the ceiling cache and the slot table are module-level —
    reset around every test so state can't leak between them."""
    usage_budget._reset_global_cache()
    usage_budget._active_streams.clear()
    yield
    usage_budget._reset_global_cache()
    usage_budget._active_streams.clear()


# ---------------------------------------------------------------------------
# check_user_budget
# ---------------------------------------------------------------------------


async def test_user_under_budget_passes(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_budget_usd_per_user", 0.50
    )
    fake_db.queue(0.10)
    await check_user_budget(fake_db, "uid-1")  # no raise


async def test_user_with_no_usage_row_passes(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_budget_usd_per_user", 0.50
    )
    # Empty queue → scalar None → first request of the day.
    await check_user_budget(fake_db, "uid-1")


async def test_user_at_budget_blocked_until_utc_midnight(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_budget_usd_per_user", 0.50
    )
    fake_db.queue(0.50)  # at budget counts as exceeded
    with pytest.raises(BudgetExceeded) as exc_info:
        await check_user_budget(fake_db, "uid-1")
    exc = exc_info.value
    assert exc.detail == "daily_budget_exceeded"
    assert exc.status_code == 429
    assert 1 <= exc.retry_after_seconds <= 86400


async def test_user_budget_zero_disables_check(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_budget_usd_per_user", 0.0
    )
    await check_user_budget(fake_db, "uid-1")
    assert fake_db.executed == []  # disabled → not even a SELECT


# ---------------------------------------------------------------------------
# check_global_ceiling
# ---------------------------------------------------------------------------


async def test_global_ceiling_blocks_with_503(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_ceiling_usd_global", 1.0
    )
    fake_db.queue(2.0)
    with pytest.raises(BudgetExceeded) as exc_info:
        await check_global_ceiling(fake_db)
    assert exc_info.value.detail == "service_at_capacity"
    assert exc_info.value.status_code == 503


async def test_global_ceiling_under_passes_and_caches(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_ceiling_usd_global", 10.0
    )
    fake_db.queue(0.5)
    await check_global_ceiling(fake_db)
    assert len(fake_db.executed) == 1
    # Second call within the cache window must not hit the DB again.
    await check_global_ceiling(fake_db)
    assert len(fake_db.executed) == 1


async def test_global_ceiling_zero_disables_check(monkeypatch, fake_db):
    monkeypatch.setattr(
        usage_budget.settings, "daily_cost_ceiling_usd_global", 0.0
    )
    await check_global_ceiling(fake_db)
    assert fake_db.executed == []


# ---------------------------------------------------------------------------
# record_usage
# ---------------------------------------------------------------------------


async def test_record_usage_upserts_and_commits(fake_db):
    await record_usage(
        fake_db,
        "uid-1",
        input_tokens=100,
        output_tokens=200,
        cache_creation_input_tokens=3000,
        cache_read_input_tokens=1500,
        estimated_cost_usd=0.00445,
    )
    assert len(fake_db.executed) == 1
    stmt = fake_db.executed[0]
    assert stmt.table.name == "usage_daily"
    assert fake_db.committed == 1


async def test_record_usage_is_fail_soft():
    """A broken accounting write must never propagate into the stream."""

    class BoomSession:
        rolled_back = 0

        async def execute(self, _stmt):
            raise RuntimeError("db exploded")

        async def rollback(self):
            self.rolled_back += 1

    session = BoomSession()
    await record_usage(  # must not raise
        session,
        "uid-1",
        input_tokens=1,
        output_tokens=1,
        estimated_cost_usd=0.0,
    )
    assert session.rolled_back == 1


# ---------------------------------------------------------------------------
# Concurrent stream slots
# ---------------------------------------------------------------------------


def test_slots_cap_then_release(monkeypatch):
    monkeypatch.setattr(
        usage_budget.settings, "max_concurrent_streams_per_user", 2
    )
    assert acquire_stream_slot("uid-1") is True
    assert acquire_stream_slot("uid-1") is True
    assert acquire_stream_slot("uid-1") is False
    # Other users are unaffected.
    assert acquire_stream_slot("uid-2") is True
    release_stream_slot("uid-1")
    assert acquire_stream_slot("uid-1") is True


def test_slot_ttl_self_heals_leaked_slots(monkeypatch):
    """A generator Starlette never iterated leaks its slot; the TTL prune
    reclaims it instead of locking the user out forever."""
    monkeypatch.setattr(
        usage_budget.settings, "max_concurrent_streams_per_user", 1
    )
    stale = time.monotonic() - usage_budget._STREAM_SLOT_TTL_SECONDS - 1
    usage_budget._active_streams["uid-1"] = [stale]
    assert acquire_stream_slot("uid-1") is True


def test_slot_limit_zero_disables_cap(monkeypatch):
    monkeypatch.setattr(
        usage_budget.settings, "max_concurrent_streams_per_user", 0
    )
    for _ in range(10):
        assert acquire_stream_slot("uid-1") is True


def test_release_unknown_user_is_noop():
    release_stream_slot("never-acquired")  # must not raise
