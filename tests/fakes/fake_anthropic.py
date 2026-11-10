"""Test doubles for the Anthropic streaming surface the chat agent reads.

`app/agents/chat_agent.py` consumes:

    async with client.messages.stream(...) as response:
        async for event in response:
            event.type == "text"        -> event.text
            event.type == "input_json"  -> event.snapshot   # partial input dict
        final = await response.get_final_message()
        # reads final.usage.{input,output,cache_creation,cache_read}_tokens
        # reads final.stop_reason
        # reads final.content (list of TextBlock | ToolUseBlock)

These builders recreate exactly that surface from plain data — no real SDK,
no network. `fake_anthropic_client` builds a `client`-shaped object the
chat-agent tests monkeypatch onto `chat_agent.client`; `make_stream` loads a
JSON cassette and returns the `messages.stream` factory the `conftest`
fixture installs.

Why a cassette format rather than a per-test stub: the SDK shape is fiddly
(text deltas, partial tool-input snapshots, cache token counts), so we want a
single recording schema every chat-agent test can target. A low-level HTTP
mock would re-test the SDK rather than our code; cassettes target the
post-SDK shape directly.

Cassette schema:
    {
        "deltas": ["Hello", " world", "!"],     # streamed text chunks
        "tool_uses": [{"name": "...", "input": {...}}],
        "usage": {"input_tokens": .., "output_tokens": ..,
                  "cache_creation_input_tokens": ..,
                  "cache_read_input_tokens": ..},
        "stop_reason": "tool_use"
    }
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_CASSETTES_DIR = Path(__file__).parent / "cassettes"


@dataclass
class _FakeUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int | None = None
    cache_read_input_tokens: int | None = None


@dataclass
class _FakeTextBlock:
    text: str
    type: str = "text"


@dataclass
class _FakeToolUseBlock:
    name: str
    input: dict[str, Any]
    id: str = "toolu_fake"
    type: str = "tool_use"


@dataclass
class _FakeFinalMessage:
    content: list[Any] = field(default_factory=list)
    usage: _FakeUsage = field(default_factory=_FakeUsage)
    stop_reason: str = "end_turn"


def usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_creation_input_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> _FakeUsage:
    """Build a `Message.usage`-shaped object.

    `input_tokens` already EXCLUDES cached tokens, so these four numbers are
    simply additive — a cassette states them directly, with no reconstruction.
    """
    return _FakeUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        cache_creation_input_tokens=cache_creation_input_tokens,
        cache_read_input_tokens=cache_read_input_tokens,
    )


def _growing_snapshots(tool_input: dict[str, Any]) -> list[dict[str, Any]]:
    """Return progressively-more-complete views of an `emit_itinerary` input.

    The real SDK rebuilds `content_block.input` from the accumulated partial
    JSON on every `input_json_delta`, so the agent sees the stop list grow one
    entry at a time. Emitting a single complete snapshot would leave the
    agent's "all stops except the last are safe to emit" logic (`flat[:-1]`)
    untested, so we synthesize one snapshot per stop: snapshot *k* carries the
    first *k* stops overall, with any day that ends up empty dropped.

    Anything that isn't the expected `{"days": [{"stops": [...]}, ...]}` shape
    falls back to a single complete snapshot.
    """
    days = tool_input.get("days")
    if not isinstance(days, list):
        return [tool_input]
    total = sum(
        len(d["stops"])
        for d in days
        if isinstance(d, dict) and isinstance(d.get("stops"), list)
    )
    if total <= 1:
        return [tool_input]

    snapshots: list[dict[str, Any]] = []
    for k in range(1, total + 1):
        snap = copy.deepcopy(tool_input)
        remaining = k
        kept_days = []
        for day in snap.get("days", []):
            if not isinstance(day, dict) or not isinstance(day.get("stops"), list):
                continue
            if remaining <= 0:
                break
            day["stops"] = day["stops"][:remaining]
            remaining -= len(day["stops"])
            kept_days.append(day)
        snap["days"] = kept_days
        snapshots.append(snap)
    return snapshots


class _FakeResponse:
    """Stand-in for the SDK's `MessageStream` async context return value."""

    def __init__(self, deltas: list[str], final: _FakeFinalMessage):
        self._deltas = deltas
        self._final = final

    @property
    def text_stream(self):
        # `text_stream` is an async iterable exposed as an attribute (not a
        # method). Mirror that by returning a fresh async generator on access.
        async def _gen():
            for chunk in self._deltas:
                yield chunk

        return _gen()

    async def __aiter__(self):
        # The SDK's full event stream: a `text` event per delta, then a run of
        # `input_json` events per tool_use carrying a growing input snapshot,
        # so the agent's per-stop live-map path is exercised the way the real
        # partial-JSON accumulation drives it.
        for chunk in self._deltas:
            yield SimpleNamespace(type="text", text=chunk, snapshot=chunk)
        for block in self._final.content:
            if getattr(block, "type", None) != "tool_use":
                continue
            for snap in _growing_snapshots(block.input):
                yield SimpleNamespace(
                    type="input_json", partial_json="", snapshot=snap
                )

    async def get_final_message(self):
        return self._final


class FakeAnthropicStream:
    """Async context manager that mimics `client.messages.stream(...)`.

    Construct from a cassette dict, or via `load_cassette(name)`. The returned
    `_FakeResponse` exposes the same event-stream / `text_stream` /
    `get_final_message()` surface the chat agent reads.
    """

    def __init__(self, cassette: dict[str, Any]):
        self._cassette = cassette

    async def __aenter__(self) -> _FakeResponse:
        deltas: list[str] = list(self._cassette.get("deltas", []))
        assistant_text = "".join(deltas)
        content: list[Any] = []
        if assistant_text:
            content.append(_FakeTextBlock(text=assistant_text))
        for tu in self._cassette.get("tool_uses", []):
            content.append(_FakeToolUseBlock(name=tu["name"], input=tu["input"]))
        u = self._cassette.get("usage", {})
        final = _FakeFinalMessage(
            content=content,
            usage=usage(
                input_tokens=u.get("input_tokens", 0),
                output_tokens=u.get("output_tokens", 0),
                cache_creation_input_tokens=u.get("cache_creation_input_tokens", 0),
                cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
            ),
            stop_reason=self._cassette.get("stop_reason", "end_turn"),
        )
        return _FakeResponse(deltas=deltas, final=final)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def make_stream_from_final(text_chunks: list[str], final_message: Any):
    """Return a `messages.stream`-shaped callable that replays `text_chunks`
    then the tool calls / usage / stop_reason carried on `final_message`.

    `final_message` is any object with `.content` / `.usage` / `.stop_reason`
    — the Anthropic-flavored `SimpleNamespace` carriers the router tests build
    are exactly that shape.
    """

    def _factory(**_kwargs):
        # Ignore stream kwargs (model, max_tokens, system, messages, tools);
        # the caller already decided the response.
        return _FakeStreamFromFinal(text_chunks, final_message)

    return _factory


class _FakeStreamFromFinal:
    """`FakeAnthropicStream` for a pre-built final message (no cassette)."""

    def __init__(self, deltas: list[str], final: Any):
        self._deltas = list(deltas)
        self._final = final

    async def __aenter__(self) -> _FakeResponse:
        return _FakeResponse(deltas=self._deltas, final=self._final)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


def fake_anthropic_client(
    text_chunks: list[str], final_message: Any
) -> SimpleNamespace:
    """Build a `client`-shaped namespace whose `client.messages.stream(...)`
    streams `text_chunks` then the tool calls + usage on `final_message`."""
    return SimpleNamespace(
        messages=SimpleNamespace(
            stream=make_stream_from_final(text_chunks, final_message)
        )
    )


def load_cassette(name: str) -> dict[str, Any]:
    """Load a cassette by stem (no `.json`)."""
    path = _CASSETTES_DIR / f"{name}.json"
    if not path.exists():
        available = sorted(p.stem for p in _CASSETTES_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"cassette {name!r} not found; available: {available}"
        )
    return json.loads(path.read_text())


def make_stream(name: str):
    """Load a cassette and return the `messages.stream` factory the
    `fake_anthropic_stream` fixture (conftest.py) installs on the shared
    client. Tests that need to inspect / mutate a cassette call
    `load_cassette` directly."""
    cassette = load_cassette(name)

    def _factory(**_kwargs):
        return FakeAnthropicStream(cassette)

    return _factory
