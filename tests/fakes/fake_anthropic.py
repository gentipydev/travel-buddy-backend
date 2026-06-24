"""Test double for `client.beta.prompt_caching.messages.stream(...)`.

The chat agent uses the Anthropic SDK's streaming context manager:

    async with client.beta.prompt_caching.messages.stream(...) as response:
        async for text in response.text_stream:
            yield encode_sse_text(text)
        final = await response.get_final_message()
        # reads final.usage.{input,output,cache_*}_tokens
        # reads final.content (list of TextBlock | ToolUseBlock)

`FakeAnthropicStream` recreates exactly that surface from a JSON cassette,
without holding the real SDK or making any network calls. Cassettes live
under `tests/fakes/cassettes/*.json` and are loaded by `load_cassette`.

Why a cassette format rather than a per-test stub: the SDK shape is
fiddly (text deltas, tool_use blocks, cache token counts), so we want a
single recording schema that every chat-agent test can target. The
`responses` library does this for HTTP; Anthropic's SDK has its own
parsing layer so a low-level HTTP mock would re-test the SDK rather
than our code. Cassettes target the post-SDK shape directly.
"""
from __future__ import annotations

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


class _FakeResponse:
    """Stand-in for the SDK's `MessageStream` async context return value."""

    def __init__(self, deltas: list[str], final: _FakeFinalMessage):
        self._deltas = deltas
        self._final = final

    @property
    def text_stream(self):
        # `text_stream` is an async iterable; the SDK exposes it as an
        # attribute (not a method). Mirror that by returning a fresh async
        # generator on each access.
        async def _gen():
            for chunk in self._deltas:
                yield chunk

        return _gen()

    async def __aiter__(self):
        # The SDK's full event stream: a `text` event per delta, then one
        # `input_json` event (full snapshot) per emit_itinerary tool_use,
        # so the agent's per-stop live-map path is exercised.
        for chunk in self._deltas:
            yield SimpleNamespace(type="text", text=chunk, snapshot=chunk)
        for block in self._final.content:
            if getattr(block, "type", None) == "tool_use" and (
                getattr(block, "name", None) == "emit_itinerary"
            ):
                yield SimpleNamespace(type="input_json", snapshot=block.input)

    async def get_final_message(self):
        return self._final


class FakeAnthropicStream:
    """Async context manager that mimics `client.beta.prompt_caching.messages.stream`.

    Construct directly from a cassette dict or via `load_cassette(name)`.
    The returned `_FakeResponse` exposes the same `text_stream` /
    `get_final_message()` surface the chat agent reads.

    Cassette schema (see `cassettes/*.json` for examples):
        {
            "deltas": ["Hello", " world", "!"],
            "tool_uses": [
                {"name": "emit_itinerary", "input": {...}}
            ],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 8,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 90
            },
            "stop_reason": "tool_use"
        }
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
        usage_dict = self._cassette.get("usage", {})
        final = _FakeFinalMessage(
            content=content,
            usage=_FakeUsage(
                input_tokens=usage_dict.get("input_tokens", 0),
                output_tokens=usage_dict.get("output_tokens", 0),
                cache_creation_input_tokens=usage_dict.get("cache_creation_input_tokens"),
                cache_read_input_tokens=usage_dict.get("cache_read_input_tokens"),
            ),
            stop_reason=self._cassette.get("stop_reason", "end_turn"),
        )
        return _FakeResponse(deltas=deltas, final=final)

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None


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
    """Convenience: load a cassette and return a callable returning a
    `FakeAnthropicStream` (the shape `client.beta.prompt_caching.messages.stream`
    has — a callable that returns an async context manager).

    Used in `conftest.py`'s `fake_anthropic_stream` fixture; tests that
    need to inspect / mutate the cassette can call `load_cassette` directly.
    """
    cassette = load_cassette(name)

    def _factory(**_kwargs):
        # Ignore stream kwargs (model, max_tokens, system, messages, tools);
        # the cassette already encodes the response.
        return FakeAnthropicStream(cassette)

    return _factory
