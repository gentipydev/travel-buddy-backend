"""Router-level SSE integration tests for /chat/stream.

These tests pin the exact SSE event ordering produced by
`app/agents/chat_agent.py`. The Flutter parser closes the SSE connection
on `data: [DONE]\\n\\n` and treats anything after as silently lost — so
the regression net here is "every byte must hit the wire BEFORE [DONE],
in the documented order."

Mocking strategy
----------------
- `prepare_turn` (the upstream context builder) is monkey-patched on the
  chat router to return a hand-built `ChatTurnContext`, skipping every
  DB / RAG / Cohere call.
- `client` is monkey-patched on the chat_agent module so
  `client.beta.prompt_caching.messages.stream(...)` returns a fake async
  context manager that yields chosen text chunks via `text_stream` and a
  chosen final message via `get_final_message()`.
- FastAPI dependency overrides bypass `get_optional_user` and
  `get_session` so no real auth/DB session is needed.

What is covered today
---------------------
The anonymous, no-trip path: plain text, conversation_id ordering,
place_card ordering, valid emit_itinerary, invalid emit_itinerary
(`event: itinerary_invalid`), the outer-try error branch
(`event: error`, no DONE), and `RequestTrace` cache-token capture.

Deferred (need a faker for SQLAlchemy session + Trip + replace/refine
helpers): refine-emit replacement, refine patch-tool path, and the
`event: persist_failed` branch.
"""
from __future__ import annotations

import json
import logging
from types import SimpleNamespace
from typing import Any
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.agents import chat_agent as chat_agent_module
from app.auth.supabase_auth import get_optional_user
from app.db.engine import get_session
from app.domain.chat_pipeline import ChatTurnContext
from app.main import app
from app.routers import chat as chat_router_module


# ---------------------------------------------------------------------------
# Fakes that quack like Anthropic types.
# ---------------------------------------------------------------------------
# The chat agent only touches a small surface of the Anthropic SDK types:
#   - `response.text_stream` async-iterable of strings
#   - `response.get_final_message()` returning a Message with
#     `.usage.input_tokens` / `.output_tokens` /
#     `.cache_creation_input_tokens` / `.cache_read_input_tokens`,
#     `.content` (list of text / tool_use blocks), `.stop_reason`.
# SimpleNamespace duck-types all of it.


def _text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def _tool_use(name: str, input_: dict) -> SimpleNamespace:
    return SimpleNamespace(type="tool_use", name=name, input=input_)


def _usage(
    *,
    input_tokens: int = 100,
    output_tokens: int = 200,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> SimpleNamespace:
    return SimpleNamespace(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def _final_message(
    content: list[SimpleNamespace],
    *,
    stop_reason: str = "end_turn",
    usage: SimpleNamespace | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=content, stop_reason=stop_reason, usage=usage or _usage(),
    )


class _FakeStreamResponse:
    """Quacks like `client.beta.prompt_caching.messages.stream(...)`'s yielded response."""

    def __init__(self, text_chunks: list[str], final_message: Any):
        self._chunks = text_chunks
        self._final = final_message

    @property
    def text_stream(self):
        async def _gen():
            for chunk in self._chunks:
                yield chunk

        return _gen()

    async def get_final_message(self):
        return self._final


class _FakeStreamCM:
    """Async context manager wrapper — matches the SDK's stream() return shape."""

    def __init__(self, response: _FakeStreamResponse):
        self._response = response

    async def __aenter__(self):
        return self._response

    async def __aexit__(self, exc_type, exc_val, tb):
        return False


def _fake_client(text_chunks: list[str], final_message: Any) -> SimpleNamespace:
    """Build a `client`-shaped namespace whose
    `client.beta.prompt_caching.messages.stream(...)` returns our fake CM."""

    def _stream(*_args, **_kwargs):
        return _FakeStreamCM(_FakeStreamResponse(text_chunks, final_message))

    return SimpleNamespace(
        beta=SimpleNamespace(
            prompt_caching=SimpleNamespace(
                messages=SimpleNamespace(stream=_stream),
            ),
        ),
    )


# ---------------------------------------------------------------------------
# SSE parser — the inverse of `app/llm/streaming.py:encode_sse_text`.
# ---------------------------------------------------------------------------


def parse_sse(body: str) -> list[tuple[str | None, str]]:
    """Split an SSE response body into ordered `(event, data)` tuples.

    `event` is `None` for default-type frames (`data: ...\\n\\n` only).
    Multi-line `data:` segments inside one frame are joined back with
    newlines (the inverse of `encode_sse_text`).
    """
    frames: list[tuple[str | None, str]] = []
    for chunk in body.split("\n\n"):
        if not chunk.strip():
            continue
        event: str | None = None
        data_lines: list[str] = []
        for line in chunk.split("\n"):
            if line.startswith("event: "):
                event = line[len("event: "):]
            elif line.startswith("data: "):
                data_lines.append(line[len("data: "):])
        frames.append((event, "\n".join(data_lines)))
    return frames


def event_sequence(frames: list[tuple[str | None, str]]) -> list[str]:
    """Sequence of event names, with `data` standing in for default frames."""
    return [e if e is not None else "data" for e, _ in frames]


# ---------------------------------------------------------------------------
# Fixtures.
# ---------------------------------------------------------------------------


@pytest.fixture
def client_anon():
    """FastAPI TestClient with anonymous auth + no-op DB session.

    Both dependencies are overridden so the agent runs against `db=None`
    and `supa_user=None`. The agent never touches either on the
    no-refine, no-conversation path that these tests exercise.
    """

    def _no_user():
        return None

    def _no_db():
        return None

    app.dependency_overrides[get_optional_user] = _no_user
    app.dependency_overrides[get_session] = _no_db
    try:
        with TestClient(app) as tc:
            yield tc
    finally:
        app.dependency_overrides.pop(get_optional_user, None)
        app.dependency_overrides.pop(get_session, None)


def _ctx(
    *,
    cached_prefix: str = "BASE",
    volatile_suffix: str = "",
    messages: list[dict] | None = None,
    new_conversation_id: Any = None,
    place_card_payload: str | None = None,
    refine_trip: Any = None,
) -> ChatTurnContext:
    return ChatTurnContext(
        cached_prefix=cached_prefix,
        volatile_suffix=volatile_suffix,
        messages=messages or [{"role": "user", "content": "hi"}],
        conversation=None,
        new_conversation_id=new_conversation_id,
        place_card_payload=place_card_payload,
        rag_sources_payload=None,
        retrieval_used=False,
        chunks_returned=0,
        top_rerank_score=None,
        refine_trip=refine_trip,
    )


def _patch_prepare_turn(monkeypatch, ctx: ChatTurnContext) -> None:
    """Replace the chat router's `prepare_turn` reference."""

    async def _fake_prepare(_request, _supa_user, _db):
        return ctx

    monkeypatch.setattr(chat_router_module, "prepare_turn", _fake_prepare)


def _patch_client(monkeypatch, text_chunks: list[str], final_message: Any) -> None:
    """Replace `client` in the chat_agent module."""
    monkeypatch.setattr(
        chat_agent_module, "client", _fake_client(text_chunks, final_message)
    )


# ---------------------------------------------------------------------------
# Tests — frame ordering invariants.
# ---------------------------------------------------------------------------


def test_plain_text_turn_emits_data_then_done(monkeypatch, client_anon):
    """No tools, no conversation: only `data:` text frames + `data: [DONE]`."""
    _patch_prepare_turn(monkeypatch, _ctx())
    _patch_client(
        monkeypatch,
        text_chunks=["Hello", " there!"],
        final_message=_final_message(content=[_text_block("Hello there!")]),
    )

    response = client_anon.post("/chat/stream", json={"message": "hi"})
    assert response.status_code == 200

    frames = parse_sse(response.text)
    assert event_sequence(frames) == ["data", "data", "data"]
    assert frames[0] == (None, "Hello")
    assert frames[1] == (None, " there!")
    assert frames[-1] == (None, "[DONE]")


def test_conversation_id_event_lands_first(monkeypatch, client_anon):
    """When `prepare_turn` produces a new_conversation_id, that frame ships
    FIRST — before any text or [DONE]. The Flutter client uses this to
    pin the new conversation in its store before the first token lands.
    """
    cid = uuid4()
    _patch_prepare_turn(monkeypatch, _ctx(new_conversation_id=cid))
    _patch_client(
        monkeypatch,
        text_chunks=["ok"],
        final_message=_final_message(content=[_text_block("ok")]),
    )

    response = client_anon.post("/chat/stream", json={"message": "hi"})
    frames = parse_sse(response.text)
    assert frames[0] == ("conversation_id", str(cid))
    assert event_sequence(frames) == ["conversation_id", "data", "data"]
    assert frames[-1] == (None, "[DONE]")


def test_place_card_event_precedes_text(monkeypatch, client_anon):
    """A place_card frame, when present, lands after conversation_id (if
    any) and before any text chunks."""
    payload = json.dumps({"name": "Tirana", "id": "1"})
    _patch_prepare_turn(monkeypatch, _ctx(place_card_payload=payload))
    _patch_client(
        monkeypatch,
        text_chunks=["ok"],
        final_message=_final_message(content=[_text_block("ok")]),
    )

    response = client_anon.post(
        "/chat/stream", json={"message": "tell me about Tirana"}
    )
    frames = parse_sse(response.text)
    assert frames[0] == ("place_card", payload)
    assert frames[1] == (None, "ok")
    assert frames[-1] == (None, "[DONE]")


def test_valid_itinerary_emits_itinerary_event(monkeypatch, client_anon):
    """A valid emit_itinerary tool call lands as `event: itinerary` BEFORE
    [DONE], with the validated payload as JSON in the data field."""
    itinerary_input = {
        "title": "Quick Tirana",
        "days": [
            {
                "day_number": 1,
                "stops": [{"name": "Skanderbeg Square", "kind": "sight"}],
            }
        ],
    }
    _patch_prepare_turn(monkeypatch, _ctx())
    _patch_client(
        monkeypatch,
        text_chunks=["Here's", " a plan."],
        final_message=_final_message(
            content=[
                _text_block("Here's a plan."),
                _tool_use("emit_itinerary", itinerary_input),
            ],
            stop_reason="tool_use",
        ),
    )

    response = client_anon.post(
        "/chat/stream", json={"message": "plan me a day in Tirana"}
    )
    frames = parse_sse(response.text)
    seq = event_sequence(frames)
    # Text chunks → itinerary → [DONE].
    assert seq[-2] == "itinerary"
    assert frames[-1] == (None, "[DONE]")

    itin = json.loads(frames[-2][1])
    assert itin["title"] == "Quick Tirana"
    assert len(itin["days"]) == 1
    assert itin["days"][0]["stops"][0]["name"] == "Skanderbeg Square"


def test_invalid_itinerary_emits_itinerary_invalid(monkeypatch, client_anon):
    """emit_itinerary with a bad shape (missing `days`) becomes
    `event: itinerary_invalid` carrying the Pydantic error summary —
    NOT `event: itinerary`. The natural-language reply still streamed
    normally before this frame.
    """
    _patch_prepare_turn(monkeypatch, _ctx())
    _patch_client(
        monkeypatch,
        text_chunks=["Here's a plan."],
        final_message=_final_message(
            content=[
                _text_block("Here's a plan."),
                # `days` missing — TripCreate.model_validate rejects.
                _tool_use("emit_itinerary", {"title": "Broken"}),
            ],
            stop_reason="tool_use",
        ),
    )

    response = client_anon.post(
        "/chat/stream", json={"message": "plan me a trip"}
    )
    frames = parse_sse(response.text)
    seq = event_sequence(frames)
    assert "itinerary" not in seq, "valid itinerary event must not appear"
    assert seq[-2] == "itinerary_invalid"
    assert frames[-1] == (None, "[DONE]")

    err = json.loads(frames[-2][1])
    assert "reason" in err
    assert "days" in err["reason"]  # Pydantic's missing-field message
    assert err["stop_reason"] == "tool_use"
    # `raw_keys` is the sorted list of keys present in Claude's input —
    # a cheap diagnostic that survives across SDK versions.
    assert err["raw_keys"] == ["title"]


def test_stream_error_emits_error_event_no_done(monkeypatch, client_anon):
    """When the Anthropic call raises synchronously, the outer try/except
    yields `event: error\\ndata: <exc message>\\n\\n` and falls through
    without yielding `[DONE]`. The Flutter client treats absence of
    [DONE] + presence of `error` as a request-level failure."""
    _patch_prepare_turn(monkeypatch, _ctx())

    def _explode(*_args, **_kwargs):
        raise RuntimeError("anthropic exploded")

    fake = SimpleNamespace(
        beta=SimpleNamespace(
            prompt_caching=SimpleNamespace(
                messages=SimpleNamespace(stream=_explode),
            ),
        ),
    )
    monkeypatch.setattr(chat_agent_module, "client", fake)

    response = client_anon.post("/chat/stream", json={"message": "hi"})
    frames = parse_sse(response.text)
    seq = event_sequence(frames)

    # The error frame must be present. [DONE] must NOT be — the success
    # path yields it just before the trace log; the except branch
    # bypasses that line.
    assert "error" in seq
    assert "[DONE]" not in (data for _, data in frames)
    err_data = next(d for e, d in frames if e == "error")
    assert "anthropic exploded" in err_data


# ---------------------------------------------------------------------------
# Tests — telemetry capture.
# ---------------------------------------------------------------------------


def test_cache_tokens_recorded_on_request_trace(monkeypatch, client_anon, caplog):
    """`cache_creation_input_tokens` and `cache_read_input_tokens` ride on
    `final.usage` and are emitted on the `RequestTrace` JSON log line.
    This is what dashboards key off to compute cache hit rate."""
    _patch_prepare_turn(monkeypatch, _ctx())
    _patch_client(
        monkeypatch,
        text_chunks=["ok"],
        final_message=_final_message(
            content=[_text_block("ok")],
            usage=_usage(
                input_tokens=50,
                output_tokens=100,
                cache_creation_input_tokens=3000,
                cache_read_input_tokens=1500,
            ),
        ),
    )

    with caplog.at_level(logging.INFO, logger="app.observability.telemetry"):
        response = client_anon.post("/chat/stream", json={"message": "hi"})
        assert response.status_code == 200

    trace_records = [
        r for r in caplog.records if r.name == "app.observability.telemetry"
    ]
    assert trace_records, "expected a RequestTrace JSON log record"

    trace = json.loads(trace_records[-1].getMessage())
    assert trace["cache_creation_input_tokens"] == 3000
    assert trace["cache_read_input_tokens"] == 1500
    assert trace["input_tokens"] == 50
    assert trace["output_tokens"] == 100
    assert trace["error"] is None
    assert trace["itinerary_emitted"] is False
    assert trace["trip_refine"] is False
    # Stage telemetry (docs/07 item 4). `prepare_turn` is monkey-patched
    # out, so context_assembly / rag_* are absent — anthropic_stream is
    # the stage the real ChatAgent.stream() body still runs.
    assert isinstance(trace["stage_latencies"], dict)
    assert "anthropic_stream" in trace["stage_latencies"]
    assert trace["stage_latencies"]["anthropic_stream"] > 0
    # Cost derived from the patched usage at Haiku rates:
    #   50 × $1.00/1M  + 100 × $5.00/1M
    # + 3000 × $1.25/1M + 1500 × $0.10/1M  =  0.00445
    assert trace["estimated_cost_usd"] == pytest.approx(0.00445)


def test_itinerary_emitted_flag_set_on_trace(monkeypatch, client_anon, caplog):
    """A successful emit_itinerary sets `itinerary_emitted=True` and
    captures `itinerary_days` on the telemetry trace."""
    _patch_prepare_turn(monkeypatch, _ctx())
    _patch_client(
        monkeypatch,
        text_chunks=["ok"],
        final_message=_final_message(
            content=[
                _text_block("ok"),
                _tool_use(
                    "emit_itinerary",
                    {
                        "title": "Plan",
                        "days": [
                            {"day_number": 1, "stops": [{"name": "X", "kind": "sight"}]},
                            {"day_number": 2, "stops": [{"name": "Y", "kind": "sight"}]},
                        ],
                    },
                ),
            ],
            stop_reason="tool_use",
        ),
    )

    with caplog.at_level(logging.INFO, logger="app.observability.telemetry"):
        response = client_anon.post("/chat/stream", json={"message": "plan"})
        assert response.status_code == 200

    trace_records = [
        r for r in caplog.records if r.name == "app.observability.telemetry"
    ]
    assert trace_records, "expected a RequestTrace JSON log record"
    trace = json.loads(trace_records[-1].getMessage())
    assert trace["itinerary_emitted"] is True
    assert trace["itinerary_days"] == 2
    assert trace["itinerary_validation_error"] is None
