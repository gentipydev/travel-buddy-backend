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

import asyncio
import uuid
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from app.agents import chat_agent as chat_agent_module
from app.agents.chat_agent import ChatAgent
from app.llm import gemini_client as gemini_client_module
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
        model="gemini-2.5-flash",
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
        gemini_client_module,
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
# Live-map per-stop streaming (initial-creation path).
# ---------------------------------------------------------------------------


async def test_emit_itinerary_streams_per_stop_markers(monkeypatch, fake_db):
    """On a non-refine emit_itinerary, each geocoded stop streams as an
    `itinerary_stop` frame (preceded by one `itinerary_stream_start`), the
    `itinerary` Save payload still lands, and [DONE] is last."""
    itinerary = {
        "title": "Tirana",
        "days": [
            {
                "day_number": 1,
                "title": "Old Town & history",
                "stops": [
                    {
                        "name": "Skanderbeg Square",
                        "kind": "sight",
                        "notes": "The beating heart of the city.",
                    },
                    {"name": "Bunk'Art 2", "kind": "sight"},
                ],
            }
        ],
    }
    monkeypatch.setattr(
        gemini_client_module,
        "client",
        _fake_client(
            text_chunks=["A lovely day."],
            final_message=_final_message(
                content=[
                    _text_block("A lovely day."),
                    _tool_use("emit_itinerary", itinerary),
                ],
                stop_reason="tool_use",
            ),
        ),
    )

    # Override the conftest no-op so stops actually geocode to coords.
    async def _fake_geocode(*, name, city, lat, lng):
        return (41.33, 19.82, f"pid-{name}")

    monkeypatch.setattr(chat_agent_module, "geocode_stop_preview", _fake_geocode)

    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="a day in Tirana"),
        ctx=_ctx(),  # no conversation, no refine → anonymous create path
        user_id=None,
        trace=_new_trace(),
        started_at=0.0,
    )
    frames = await _drain(agent.stream())
    events = _events(frames)

    assert events.count("itinerary_stream_start") == 1
    assert events.count("itinerary_stop") == 2  # both stops mapped
    # Live markers stream BEFORE the Save payload, which precedes [DONE].
    assert events.index("itinerary_stream_start") < events.index("itinerary")
    assert "itinerary" in events
    assert frames[-1] == "data: [DONE]\n\n"

    # Each stop frame carries usable coordinates.
    stop_frames = [f for f in frames if f.startswith("event: itinerary_stop")]
    import json as _json

    first = _json.loads(stop_frames[0].split("data: ", 1)[1])
    assert first["lat"] == 41.33 and first["lng"] == 19.82
    assert first["day_number"] == 1
    # Per-stop prose (from `notes`) and the day title ride along the live frame
    # so the chat row renders in lockstep with the marker.
    assert first["description"] == "The beating heart of the city."
    assert first["day_title"] == "Old Town & history"


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
        gemini_client_module,
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
        gemini_client_module,
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
        gemini_client_module,
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
        gemini_client_module,
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


async def test_llm_stream_stage_recorded_on_success_path(
    monkeypatch, fake_db,
):
    """The manual `_llm_start = time.perf_counter()` timer must record a
    value > 0 once the `generate_content_stream(...)` loop finishes."""
    monkeypatch.setattr(
        gemini_client_module,
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
    assert "llm_stream" in trace.stage_latencies
    assert trace.stage_latencies["llm_stream"] > 0


async def test_persistence_stage_only_records_when_conversation_present(
    monkeypatch, fake_db,
):
    """Persistence is wrapped in `with self.trace.stage("persistence")`. The
    block only enters when ctx.conversation is set, so no-conversation turns
    have no `persistence` key in stage_latencies."""
    monkeypatch.setattr(
        gemini_client_module,
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
    """compute_cost runs before log_trace and lands on self.trace. At
    gemini-2.5-flash rates: 100 input × $0.30/1M + 200 output × $2.50/1M
    = 0.00053."""
    monkeypatch.setattr(
        gemini_client_module,
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
    assert trace.estimated_cost_usd == pytest.approx(0.00053)


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


# ---------------------------------------------------------------------------
# Cassette-driven tests via `fake_gemini_stream` fixture (conftest.py).
#
# These verify the same paths as the inline `_fake_client` tests above but
# through the recorded-response cassette layer — the path the eval harness
# will exercise too. Adding tests at this layer means a cassette schema
# change is caught here before it shows up in evals/runners/braintrust_run.
# ---------------------------------------------------------------------------


async def test_chat_agent_text_only_cassette_streams_and_captures_cache_tokens(
    fake_gemini_stream, monkeypatch, fake_db,
):
    """The `chat_text_only` cassette has no tool calls and ~3120 cached
    input tokens. After the stream completes:
    - text deltas should be emitted as `data:` frames (no event prefix)
    - no itinerary or trip_refined event
    - cache_read_input_tokens should land on the trace from `response.usage`
    """
    fake_gemini_stream("chat_text_only")
    # Persistence is out of scope for this test — stub the writer that
    # would otherwise need a real conversation row.
    monkeypatch.setattr(
        chat_agent_module, "append_messages", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        chat_agent_module, "maybe_summarize", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        chat_agent_module, "generate_title", AsyncMock(return_value=None)
    )

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="is tap water safe in Berat?"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    events = _events(frames)
    assert "itinerary" not in events
    assert "trip_refined" not in events
    # At least one default text frame landed.
    assert "data" in events
    # [DONE] is the terminator.
    assert frames[-1] == "data: [DONE]\n\n"
    # The cassette's usage block landed on the trace.
    assert trace.cache_read_input_tokens == 3120
    assert trace.input_tokens == 1240
    assert trace.output_tokens == 64
    # No itinerary emitted.
    assert trace.itinerary_emitted is False


async def test_chat_agent_emit_itinerary_cassette_emits_itinerary_event(
    fake_gemini_stream, monkeypatch, fake_db,
):
    """The `chat_emit_itinerary` cassette pairs prose deltas with a valid
    `emit_itinerary` tool block (stops are in Albania so the geo validator
    passes). The chat agent should:
    - stream the prose
    - emit `event: itinerary` with the validated payload
    - record itinerary_emitted=True + days count on the trace
    """
    fake_gemini_stream("chat_emit_itinerary")
    monkeypatch.setattr(
        chat_agent_module, "append_messages", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        chat_agent_module, "maybe_summarize", AsyncMock(return_value=None)
    )
    monkeypatch.setattr(
        chat_agent_module, "generate_title", AsyncMock(return_value=None)
    )

    trace = _new_trace()
    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="2 days starting from Tirana"),
        ctx=_ctx(),
        user_id=None,
        trace=trace,
        started_at=0.0,
    )
    frames = await _drain(agent.stream())

    events = _events(frames)
    assert "itinerary" in events
    # No refine (no trip_id on the request).
    assert "trip_refined" not in events
    # itinerary_invalid would mean the geo validator rejected the cassette;
    # the cassette is hand-curated to pass.
    assert "itinerary_invalid" not in events
    assert frames[-1] == "data: [DONE]\n\n"
    assert trace.itinerary_emitted is True
    assert trace.itinerary_days == 2
    # Cache tokens flow through unchanged from the cassette.
    assert trace.cache_read_input_tokens == 3120


# ---------------------------------------------------------------------------
# Mid-stream cancellation: client disconnects while the stream is running.
# ---------------------------------------------------------------------------


async def test_midstream_cancellederror_propagates_not_swallowed(
    monkeypatch, fake_db,
):
    """A client disconnect cancels the still-running generator at its next
    await. In Python 3.12 that injects `asyncio.CancelledError` — a
    BaseException, NOT an `Exception` — so the agent's outer `except Exception`
    must NOT catch it.

    This is the safety basis for the persist-before-[DONE] invariant: if a
    cancel could be swallowed and turned into a normal return (or an
    `event: error` frame), a post-[DONE] write could silently be skipped.
    Here we throw the cancel mid-text-stream and assert it surfaces unchanged
    — no `event: error` frame, no clean completion."""
    monkeypatch.setattr(
        gemini_client_module,
        "client",
        _fake_client(
            text_chunks=["first ", "second ", "third"],
            final_message=_final_message(
                content=[_text_block("first second third")]
            ),
        ),
    )

    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hello"),
        ctx=_ctx(),
        user_id=None,
        trace=_new_trace(),
        started_at=0.0,
    )
    gen = agent.stream()
    # Pull the first frame so the generator is suspended at a text-stream
    # yield (no conversation_id / place_card on the default ctx, so the first
    # yield is a text data frame).
    first = await gen.__anext__()
    assert first.startswith("data: ")

    # Simulate Starlette cancelling the generator mid-stream. The cancel must
    # propagate out of athrow rather than being converted to a yielded frame.
    with pytest.raises(asyncio.CancelledError):
        await gen.athrow(asyncio.CancelledError())


async def test_disconnect_releases_slot_and_schedules_estimated_usage(
    monkeypatch, fake_db,
):
    """Cost-abuse hardening: when the client disconnects mid-stream, the
    wrapper's `finally` must (a) release the user's concurrency slot and
    (b) schedule a detached `usage_daily` write estimating the output
    already streamed (~4 chars/token) — partial streams cost money too.
    """
    from app.domain import usage_budget

    recorded: list[dict] = []

    async def _fake_detached(uid, **kwargs):
        recorded.append({"uid": uid, **kwargs})

    monkeypatch.setattr(
        chat_agent_module, "record_usage_detached", _fake_detached
    )
    monkeypatch.setattr(
        gemini_client_module,
        "client",
        _fake_client(
            text_chunks=["twelve chars", "never sent"],
            final_message=_final_message(
                content=[_text_block("twelve chars never sent")]
            ),
        ),
    )

    usage_budget._active_streams.clear()
    # The router acquires the slot before constructing the agent.
    assert usage_budget.acquire_stream_slot("uid-1") is True

    agent = ChatAgent(
        db=fake_db,
        request=ChatRequest(message="hello"),
        ctx=_ctx(),
        user_id="uid-1",
        trace=_new_trace(user_id="uid-1"),
        started_at=0.0,
    )
    gen = agent.stream()
    first = await gen.__anext__()
    assert first == "data: twelve chars\n\n"

    with pytest.raises(asyncio.CancelledError):
        await gen.athrow(asyncio.CancelledError())

    # Slot released by the finally — a reconnect isn't locked out.
    assert usage_budget._active_streams == {}

    # The detached record task was scheduled during unwinding; let the
    # event loop run it. Only the chunk that hit the wire is billed:
    # 12 chars // 4 = 3 estimated output tokens, input unknown → 0.
    await asyncio.sleep(0)
    assert len(recorded) == 1
    assert recorded[0]["uid"] == "uid-1"
    assert recorded[0]["input_tokens"] == 0
    assert recorded[0]["output_tokens"] == 3
    assert recorded[0]["estimated_cost_usd"] > 0
