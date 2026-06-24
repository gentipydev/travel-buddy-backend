"""Test doubles for the Gemini streaming surface the chat agent reads.

`app/agents/chat_agent.py` consumes:

    stream = await client.aio.models.generate_content_stream(...)
    async for chunk in stream:
        for cand in chunk.candidates:
            cand.finish_reason
            for part in cand.content.parts:
                part.text                       # text delta
                part.function_call.name / .args # tool call
        chunk.usage_metadata                     # on the final chunk

These builders recreate exactly that shape from plain data — no real SDK,
no network. `fake_genai_client` builds a `client`-shaped object the chat
agent tests monkeypatch onto `chat_agent.client`; `make_stream` loads a JSON
cassette and returns the async `generate_content_stream` factory the
`conftest` fixture installs.

Cassette schema (unchanged from the Anthropic-era recordings, reinterpreted
for Gemini token accounting):
    {
        "deltas": ["Hello", " world", "!"],     # streamed text chunks
        "tool_uses": [{"name": "...", "input": {...}}],
        "usage": {"input_tokens": .., "output_tokens": ..,
                  "cache_read_input_tokens": ..},
        "stop_reason": "tool_use"
    }
`prompt_token_count` is reconstructed as input + cache_read so the agent's
`extract_usage` recovers the original input / output / cache_read values.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

_CASSETTES_DIR = Path(__file__).parent / "cassettes"


def _text_part(text: str) -> SimpleNamespace:
    return SimpleNamespace(text=text, function_call=None)


def _function_call_part(name: str, args: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        text=None, function_call=SimpleNamespace(name=name, args=args)
    )


def usage_metadata(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> SimpleNamespace:
    """Build a `usage_metadata`-shaped object. `prompt_token_count` includes
    cached tokens (as the real API reports), so `extract_usage` recovers
    input = prompt - cached."""
    return SimpleNamespace(
        prompt_token_count=input_tokens + cache_read_input_tokens,
        candidates_token_count=output_tokens,
        cached_content_token_count=cache_read_input_tokens,
        thoughts_token_count=0,
        tool_use_prompt_token_count=0,
        total_token_count=input_tokens + cache_read_input_tokens + output_tokens,
    )


def build_chunks(
    *,
    text_chunks: list[str],
    tool_uses: list[dict[str, Any]],
    usage: SimpleNamespace,
    stop_reason: str | None,
) -> list[SimpleNamespace]:
    """One chunk per text delta (text parts only), then a final chunk carrying
    any function-call parts, the finish reason, and the usage metadata."""
    chunks: list[SimpleNamespace] = []
    for t in text_chunks:
        chunks.append(
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(
                        content=SimpleNamespace(parts=[_text_part(t)]),
                        finish_reason=None,
                    )
                ],
                usage_metadata=None,
            )
        )
    final_parts = [
        _function_call_part(tu["name"], tu["input"]) for tu in tool_uses
    ]
    chunks.append(
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=SimpleNamespace(parts=final_parts),
                    finish_reason=stop_reason,
                )
            ],
            usage_metadata=usage,
        )
    )
    return chunks


def make_async_stream(chunks: list[SimpleNamespace]):
    """Return an async `generate_content_stream`-shaped callable: awaiting it
    yields an async iterator over `chunks` (mirrors the real SDK contract,
    where the method is awaited before iterating)."""

    async def _factory(**_kwargs):
        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    return _factory


def fake_genai_client(
    *,
    text_chunks: list[str],
    tool_uses: list[dict[str, Any]],
    usage: SimpleNamespace,
    stop_reason: str | None,
) -> SimpleNamespace:
    """Build a `client`-shaped namespace whose
    `client.aio.models.generate_content_stream(...)` streams the given data."""
    chunks = build_chunks(
        text_chunks=text_chunks,
        tool_uses=tool_uses,
        usage=usage,
        stop_reason=stop_reason,
    )
    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content_stream=make_async_stream(chunks))
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
    """Load a cassette and return the async `generate_content_stream` factory
    the `fake_gemini_stream` fixture installs on the shared client."""
    cassette = load_cassette(name)
    u = cassette.get("usage", {})
    usage = usage_metadata(
        input_tokens=u.get("input_tokens", 0),
        output_tokens=u.get("output_tokens", 0),
        cache_read_input_tokens=u.get("cache_read_input_tokens", 0),
    )
    chunks = build_chunks(
        text_chunks=list(cassette.get("deltas", [])),
        tool_uses=list(cassette.get("tool_uses", [])),
        usage=usage,
        stop_reason=cassette.get("stop_reason"),
    )
    return make_async_stream(chunks)
