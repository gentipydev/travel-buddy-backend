"""Tests for `app.agents.chat_agent.ChatAgent.stream()`.

`test_chat_router.py` already exercises the chat agent via the FastAPI
TestClient — those tests pin the public HTTP/SSE surface. This file goes
one layer deeper and covers the paths that integration doesn't easily
reach: the refine branches (replace_trip_from_payload + apply_refine_tool),
persistence success + failure, and the trace mutations item 4 introduced.

Strategy: invoke `ChatAgent.stream()` directly (no HTTP), iterate the async
generator, collect frames, and assert on (a) what was yielded, (b) what
methods were called on the mocked db / append_messages / etc., (c) what
fields landed on `self.trace`.
"""
from __future__ import annotations

import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agents import chat_agent as chat_agent_module
from app.agents.chat_agent import ChatAgent
from app.db.models import Conversation, Trip
from app.domain.chat_pipeline import ChatTurnContext
from app.observability import RequestTrace
from app.schemas import ChatRequest
from tests.test_chat_router import (
    _fake_client,
    _final_message,
    _text_block,
    _tool_use,
    _usage,
)

pytestmark = pytest.mark.anyio


# ---------------------------------------------------------------------------
# Builders.
# ---------------------------------------------------------------------------


def _new_trace(*, user_id: str | None = None) -> RequestTrace:
    return RequestTrace(
        request_id="test-req",
        timestamp="2026-05-18T00:00:00+00:00",
        user_id=user_id,
        message_length=5,
        retrieval_used=False,
        chunks_returned=0,
        top_rerank_score=None,
        input_tokens=None,
        output_tokens=None,
        latency_ms=0.0,
        model="claude-haiku-4-5-20251001",
        error=None,
    )


def _make_trip() -> Trip:
    now = datetime.now(UTC)
    trip = Trip(
        id=uuid.uuid4(),
        user_id=1,
        title="Saved trip",
        summary=None,
        start_date=None,
        created_at=now,
        updated_at=now,
    )
    trip.days = []
    return trip


def _make_conv() -> Conversation:
    now = datetime.now(UTC)
    return Conversation(
        id=uuid.uuid4(),
        user_id=1,
        title=None,
        created_at=now,
        updated_at=now,
    )


def _ctx(
    *,
    conversation: Conversation | None = None,
    refine_trip: Trip | None = None,
    new_conversation_id: uuid.UUID | None = None,
) -> ChatTurnContext:
    return ChatTurnContext(
        cached_prefix="BASE",
        volatile_suffix="",
        messages=[{"role": "user", "content": "hi"}],
        conversation=conversation,
        new_conversation_id=new_conversation_id,
        place_card_payload=None,
        retrieval_used=False,
        chunks_returned=0,
        top_rerank_score=None,
        refine_trip=refine_trip,
    )


async def _drain(stream) -> list[str]:
    """Consume an async generator into a list of yielded strings."""
    out: list[str] = []
    async for item in stream:
        out.append(item)
    return out


def _events(frames: list[str]) -> list[str]:
    """Return the ordered list of `event:` names; 'data' for default frames.
    Same idea as test_chat_router.event_sequence but for our raw yield list."""
    names: list[str] = []
    for f in frames:
        if f.startswith("event: "):
            names.append(f.split("\n", 1)[0][len("event: "):])
        elif f.startswith("data: "):
            names.append("data")
    return names


# ---------------------------------------------------------------------------
# Refine path: emit_itinerary on a refine turn → replace_trip_from_payload.
# ---------------------------------------------------------------------------


async def test_refine_via_emit_itinerary_commits_and_emits_trip_refined(
    monkeypatch, fake_db,
):
    """Valid emit_itinerary on a refine turn:
    - replace_trip_from_payload called
    - db.commit() fires BEFORE the trip_refined yield (durability)
    - event: trip_refined frame lands, not event: itinerary
    """
    trip = _make_trip()
    updated_trip = _make_trip()
    valid_itinerary = {
        "title": "Refined plan",
        "days": [
            {"day_number": 1, "stops": [{"name": "Berat", "kind": "sight"}]}
        ],
    }

    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["here is the plan"],
            final_message=_final_message(
                content=[
                    _text_block("here is the plan"),
                    _tool_use("emit_itinerary", valid_itinerary),
                ],
            ),
        ),
    )
    replace_mock = AsyncMock(return_value=updated_trip)
    serialize_mock = AsyncMock(return_value={"id": str(updated_trip.id)})
    monkeypatch.setattr(
        chat_agent_module, "replace_trip_from_payload", replace_mock
    )
    monkeypatch.setattr(chat_agent_module, "serialize_trip", serialize_mock)

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="refine this", trip_id=trip.id),
        ctx=_ctx(refine_trip=trip),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    replace_mock.assert_awaited_once()
    assert fake_db.committed >= 1
    events = _events(frames)
    # trip_refined supersedes "itinerary" on a refine turn.
    assert "trip_refined" in events
    assert "itinerary" not in events
    # [DONE] still lands last.
    assert frames[-1] == "data: [DONE]\n\n"
    # Trace flags reflect refine success.
    assert trace.itinerary_emitted is True
    assert trace.trip_refine is True


# ---------------------------------------------------------------------------
# Refine path: apply_refine_tool patch tools.
# ---------------------------------------------------------------------------


async def test_refine_patch_via_apply_refine_tool_commits_once(
    monkeypatch, fake_db,
):
    """Patch-tool refine (e.g. update_stop): each tool block is applied via
    apply_refine_tool, the WHOLE turn lands in ONE db.commit, and one
    trip_refined event is emitted with the post-state."""
    trip = _make_trip()
    latest = _make_trip()
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["ok"],
            final_message=_final_message(
                content=[
                    _text_block("ok"),
                    _tool_use("update_stop", {"stop_id": str(uuid.uuid4()), "name": "x"}),
                    _tool_use("update_stop", {"stop_id": str(uuid.uuid4()), "name": "y"}),
                ],
            ),
        ),
    )
    apply_mock = AsyncMock(return_value=latest)
    monkeypatch.setattr(chat_agent_module, "apply_refine_tool", apply_mock)
    monkeypatch.setattr(
        chat_agent_module, "serialize_trip",
        AsyncMock(return_value={"id": str(latest.id)}),
    )

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="patch this", trip_id=trip.id),
        ctx=_ctx(refine_trip=trip),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    # Both tool blocks dispatched, single commit, single trip_refined.
    assert apply_mock.await_count == 2
    assert fake_db.committed == 1
    events = _events(frames)
    assert events.count("trip_refined") == 1
    assert trace.itinerary_emitted is True


async def test_refine_patch_failure_rolls_back_and_emits_refine_failed(
    monkeypatch, fake_db,
):
    """If any patch in the turn fails, the WHOLE turn rolls back (never
    half-applied) and `event: refine_failed` lands instead of trip_refined."""
    trip = _make_trip()
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["trying..."],
            final_message=_final_message(
                content=[
                    _text_block("trying..."),
                    _tool_use("update_stop", {"stop_id": str(uuid.uuid4()), "name": "x"}),
                ],
            ),
        ),
    )
    apply_mock = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(chat_agent_module, "apply_refine_tool", apply_mock)

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="patch", trip_id=trip.id),
        ctx=_ctx(refine_trip=trip),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    assert fake_db.rolled_back >= 1
    events = _events(frames)
    assert "refine_failed" in events
    assert "trip_refined" not in events
    # itinerary_validation_error captures the exception type so dashboards
    # can attribute failures.
    assert trace.itinerary_validation_error is not None
    assert "RuntimeError" in trace.itinerary_validation_error


# ---------------------------------------------------------------------------
# Persistence path: append_messages + generate_title + maybe_summarize.
# ---------------------------------------------------------------------------


async def test_persistence_calls_append_then_title_then_summarize(
    monkeypatch, fake_db,
):
    """Happy persistence: assistant text + a `conversation` present →
    append_messages, generate_title (only on new conversations), and
    maybe_summarize all fire. No persist_failed frame on success."""
    conv = _make_conv()
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["hello there"],
            final_message=_final_message(
                content=[_text_block("hello there")],
            ),
        ),
    )
    append_mock = AsyncMock()
    title_mock = AsyncMock()
    summarize_mock = AsyncMock()
    monkeypatch.setattr(chat_agent_module, "append_messages", append_mock)
    monkeypatch.setattr(chat_agent_module, "generate_title", title_mock)
    monkeypatch.setattr(chat_agent_module, "maybe_summarize", summarize_mock)

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(conversation=conv, new_conversation_id=conv.id),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    append_mock.assert_awaited_once()
    # generate_title fires because new_conversation_id is set.
    title_mock.assert_awaited_once()
    summarize_mock.assert_awaited_once()
    events = _events(frames)
    assert "persist_failed" not in events


async def test_persist_failed_event_emitted_before_done(monkeypatch, fake_db):
    """If append_messages raises, the agent emits `event: persist_failed`
    BEFORE [DONE] so the Flutter client can mark the assistant message as
    not-saved inline (per memory `feedback_no_snackbar`). Critical that
    persist_failed lands BEFORE [DONE] — Starlette cancels the generator
    on [DONE] and any post-[DONE] yields silently never happen."""
    conv = _make_conv()
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["hi"],
            final_message=_final_message(content=[_text_block("hi")]),
        ),
    )
    monkeypatch.setattr(
        chat_agent_module, "append_messages",
        AsyncMock(side_effect=RuntimeError("db down")),
    )
    # title / summarize won't fire because append raises first.
    monkeypatch.setattr(chat_agent_module, "generate_title", AsyncMock())
    monkeypatch.setattr(chat_agent_module, "maybe_summarize", AsyncMock())

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(conversation=conv),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    events = _events(frames)
    assert "persist_failed" in events
    # persist_failed lands BEFORE [DONE].
    persist_idx = next(
        i for i, f in enumerate(frames) if f.startswith("event: persist_failed")
    )
    done_idx = next(
        i for i, f in enumerate(frames) if f == "data: [DONE]\n\n"
    )
    assert persist_idx < done_idx


# ---------------------------------------------------------------------------
# Item-4 stage telemetry — verify the new fields land on self.trace.
# ---------------------------------------------------------------------------


async def test_anthropic_stream_stage_recorded_on_success_path(
    monkeypatch, fake_db,
):
    """The manual `_anthropic_start = time.perf_counter()` timer must record
    a value > 0 once `async with client.beta.messages.stream(...)` closes."""
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["hi"],
            final_message=_final_message(content=[_text_block("hi")]),
        ),
    )
    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    await _drain(agent.stream())
    assert "anthropic_stream" in trace.stage_latencies
    assert trace.stage_latencies["anthropic_stream"] > 0


async def test_persistence_stage_only_records_when_conversation_present(
    monkeypatch, fake_db,
):
    """Persistence is wrapped in `with self.trace.stage("persistence")`. The
    block only enters when ctx.conversation is set, so no-conversation turns
    have no `persistence` key in stage_latencies."""
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["hi"],
            final_message=_final_message(content=[_text_block("hi")]),
        ),
    )
    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(conversation=None),  # anonymous-ish turn
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    await _drain(agent.stream())
    assert "persistence" not in trace.stage_latencies


async def test_estimated_cost_usd_derived_from_token_usage(
    monkeypatch, fake_db,
):
    """compute_cost runs before log_trace and lands on self.trace. At Haiku
    rates: 100 input × $1/1M + 200 output × $5/1M = 0.0011."""
    monkeypatch.setattr(
        chat_agent_module,
        "client",
        _fake_client(
            text_chunks=["hi"],
            final_message=_final_message(
                content=[_text_block("hi")],
                usage=_usage(input_tokens=100, output_tokens=200),
            ),
        ),
    )
    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    await _drain(agent.stream())
    assert trace.estimated_cost_usd == pytest.approx(0.0011)


# ---------------------------------------------------------------------------
# __init__ backfill — ctx-derived fields land on the trace at construction.
# ---------------------------------------------------------------------------


def test_chat_agent_init_backfills_ctx_fields_on_trace(fake_db):
    """Router constructs the trace BEFORE prepare_turn, so ctx-derived fields
    (retrieval_used, chunks_returned, top_rerank_score, trip_refine) are
    filled in by ChatAgent.__init__ — verify the assignment happens."""
    trace = _new_trace()
    ctx = SimpleNamespace(
        retrieval_used=True,
        chunks_returned=3,
        top_rerank_score=0.87,
        refine_trip=_make_trip(),
        # the rest aren't read by __init__
        new_conversation_id=None,
        place_card_payload=None,
        conversation=None,
        messages=[],
        cached_prefix="",
        volatile_suffix="",
    )
    ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hi"),
        ctx=ctx,  # type: ignore[arg-type]
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    assert trace.retrieval_used is True
    assert trace.chunks_returned == 3
    assert trace.top_rerank_score == 0.87
    assert trace.trip_refine is True
