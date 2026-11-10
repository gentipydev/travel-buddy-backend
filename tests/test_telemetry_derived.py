"""Tests for derived RequestTrace fields computed at emit time.

Two features land here:
  - `cache_hit_ratio` (item B10): cache_read / (read + creation + input),
    computed by `log_trace` just before emit, divide-by-zero guarded.
  - `output_guard_failed_open` (item B6): set by the output guard when it
    fails open (timeout / error), surfaced on the trace so a guard outage
    is no longer invisible.
"""
from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock

import pytest

from app.observability import trace_var
from app.observability.telemetry import (
    RequestTrace,
    _compute_cache_hit_ratio,
    log_trace,
)

pytestmark = pytest.mark.anyio


def _new_trace(**overrides) -> RequestTrace:
    base = dict(
        request_id="t",
        timestamp="2026-05-29T00:00:00+00:00",
        user_id=None,
        message_length=0,
        retrieval_used=False,
        chunks_returned=0,
        top_rerank_score=None,
        input_tokens=None,
        output_tokens=None,
        latency_ms=0.0,
        model="claude-haiku-4-5",
        error=None,
    )
    base.update(overrides)
    return RequestTrace(**base)


# --------------------------------------------------------------------------- #
# cache_hit_ratio                                                              #
# --------------------------------------------------------------------------- #


def test_cache_hit_ratio_computed_from_tokens():
    trace = _new_trace(
        input_tokens=100,
        cache_creation_input_tokens=300,
        cache_read_input_tokens=600,
    )
    # 600 / (600 + 300 + 100) = 0.6
    assert _compute_cache_hit_ratio(trace) == pytest.approx(0.6)


def test_cache_hit_ratio_none_when_any_token_missing():
    assert _compute_cache_hit_ratio(_new_trace()) is None
    assert (
        _compute_cache_hit_ratio(
            _new_trace(input_tokens=100, cache_read_input_tokens=50)
            # cache_creation_input_tokens still None
        )
        is None
    )


def test_cache_hit_ratio_none_on_zero_denominator():
    """All-zero token counts (a degenerate fallback path) must not divide by
    zero — returns None, not a ZeroDivisionError."""
    trace = _new_trace(
        input_tokens=0,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
    )
    assert _compute_cache_hit_ratio(trace) is None


def test_log_trace_populates_cache_hit_ratio_on_emit(caplog):
    trace = _new_trace(
        input_tokens=50,
        cache_creation_input_tokens=150,
        cache_read_input_tokens=800,
    )
    with caplog.at_level(logging.INFO, logger="app.observability.telemetry"):
        log_trace(trace)
    record = json.loads(caplog.records[-1].getMessage())
    # 800 / (800 + 150 + 50) = 0.8
    assert record["cache_hit_ratio"] == pytest.approx(0.8)
    # The field is also set in place on the trace object.
    assert trace.cache_hit_ratio == pytest.approx(0.8)


# --------------------------------------------------------------------------- #
# output_guard_failed_open                                                     #
# --------------------------------------------------------------------------- #


async def test_output_guard_sets_failed_open_flag_on_error(monkeypatch):
    """When the guard fails open (API error), it flags the ambient trace so
    the outage is visible — and still returns "safe" (contract preserved)."""
    from app.domain import output_guard as guard_module

    trace = _new_trace()
    token = trace_var.set(trace)
    try:
        monkeypatch.setattr(
            guard_module, "generate_text", AsyncMock(side_effect=Exception("api down"))
        )
        result = await guard_module.check_output_guardrail("some reply")
    finally:
        trace_var.reset(token)

    assert result == "safe"
    assert trace.output_guard_failed_open is True


async def test_output_guard_sets_failed_open_flag_on_timeout(monkeypatch):
    import asyncio

    from app.domain import output_guard as guard_module

    trace = _new_trace()
    token = trace_var.set(trace)
    try:
        monkeypatch.setattr(
            guard_module, "generate_text", AsyncMock(side_effect=asyncio.TimeoutError)
        )
        result = await guard_module.check_output_guardrail("reply")
    finally:
        trace_var.reset(token)

    assert result == "safe"
    assert trace.output_guard_failed_open is True


async def test_output_guard_does_not_flag_failed_open_on_success(monkeypatch):
    """A genuine "safe" judgment must NOT set the fail-open flag — otherwise
    the signal is useless."""
    from app.domain import output_guard as guard_module

    trace = _new_trace()
    token = trace_var.set(trace)
    try:
        monkeypatch.setattr(
            guard_module, "generate_text", AsyncMock(return_value="SAFE")
        )
        result = await guard_module.check_output_guardrail("reply")
    finally:
        trace_var.reset(token)

    assert result == "safe"
    assert trace.output_guard_failed_open is False


async def test_output_guard_failed_open_noop_without_trace(monkeypatch):
    """No ambient trace (direct unit call) → the guard still fails open
    without raising; the flag simply has nowhere to land."""
    from app.domain import output_guard as guard_module

    monkeypatch.setattr(
        guard_module, "generate_text", AsyncMock(side_effect=Exception("boom"))
    )
    # trace_var is unset here.
    result = await guard_module.check_output_guardrail("reply")
    assert result == "safe"
