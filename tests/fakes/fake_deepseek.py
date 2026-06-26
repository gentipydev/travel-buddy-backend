"""Test doubles for the DeepSeek (OpenAI-compatible) streaming surface.

`app/llm/router.py` consumes the DeepSeek fallback as:

    stream = await client.chat.completions.create(..., stream=True)
    async for chunk in stream:
        chunk.usage                              # on the final chunk
        for choice in chunk.choices:
            choice.finish_reason
            choice.delta.content                 # text delta
            choice.delta.tool_calls[].index      # fragment index
            choice.delta.tool_calls[].function.name / .arguments

These builders recreate exactly that shape from plain data — no real SDK, no
network. Tool-call `arguments` are deliberately SPLIT across deltas (keyed by
index) so the router's fragment-accumulation path is exercised, mirroring how
the real API streams function arguments as partial JSON strings.
"""
from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any


def usage(
    *,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cache_read_input_tokens: int = 0,
) -> SimpleNamespace:
    """Build a DeepSeek-shaped `usage` object. `prompt_cache_miss_tokens` is the
    full-rate input; `prompt_cache_hit_tokens` is the cheap cached read — what
    `deepseek_adapter.extract_usage` reads directly."""
    return SimpleNamespace(
        prompt_tokens=input_tokens + cache_read_input_tokens,
        completion_tokens=output_tokens,
        total_tokens=input_tokens + cache_read_input_tokens + output_tokens,
        prompt_cache_hit_tokens=cache_read_input_tokens,
        prompt_cache_miss_tokens=input_tokens,
    )


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(content=text, tool_calls=None),
            )
        ],
    )


def _tool_fragment_chunk(
    *, index: int, name: str | None, args_piece: str
) -> SimpleNamespace:
    fn = SimpleNamespace(name=name, arguments=args_piece)
    return SimpleNamespace(
        usage=None,
        choices=[
            SimpleNamespace(
                finish_reason=None,
                delta=SimpleNamespace(
                    content=None,
                    tool_calls=[SimpleNamespace(index=index, function=fn)],
                ),
            )
        ],
    )


def build_chunks(
    *,
    text_chunks: list[str],
    tool_uses: list[dict[str, Any]],
    usage_obj: SimpleNamespace,
    stop_reason: str | None,
) -> list[SimpleNamespace]:
    """One chunk per text delta, then — for each tool call — a name fragment
    followed by its JSON arguments split into two pieces (exercises
    index-keyed accumulation), then a terminal chunk carrying finish_reason,
    then a usage-only chunk (OpenAI streams usage in a final choice-less
    chunk when `stream_options.include_usage` is set)."""
    chunks: list[SimpleNamespace] = [_text_chunk(t) for t in text_chunks]
    for i, tu in enumerate(tool_uses):
        args = json.dumps(tu["input"])
        mid = len(args) // 2
        # Name arrives with the first fragment; arguments split across two.
        chunks.append(
            _tool_fragment_chunk(index=i, name=tu["name"], args_piece=args[:mid])
        )
        chunks.append(
            _tool_fragment_chunk(index=i, name=None, args_piece=args[mid:])
        )
    # Terminal choice chunk carrying the finish reason.
    chunks.append(
        SimpleNamespace(
            usage=None,
            choices=[
                SimpleNamespace(
                    finish_reason=stop_reason,
                    delta=SimpleNamespace(content=None, tool_calls=None),
                )
            ],
        )
    )
    # Usage-only final chunk (no choices).
    chunks.append(SimpleNamespace(usage=usage_obj, choices=[]))
    return chunks


def make_async_stream(chunks: list[SimpleNamespace]):
    """Return an async `chat.completions.create`-shaped callable: awaiting it
    yields an async iterator over `chunks` (mirrors the real SDK contract,
    where the coroutine is awaited before iterating the stream)."""

    async def _factory(**_kwargs):
        async def _gen():
            for chunk in chunks:
                yield chunk

        return _gen()

    return _factory


def fake_deepseek_client(
    *,
    text_chunks: list[str],
    tool_uses: list[dict[str, Any]] | None = None,
    usage_obj: SimpleNamespace | None = None,
    stop_reason: str | None = "stop",
) -> SimpleNamespace:
    """Build a `client`-shaped namespace whose
    `client.chat.completions.create(...)` streams the given data."""
    chunks = build_chunks(
        text_chunks=text_chunks,
        tool_uses=tool_uses or [],
        usage_obj=usage_obj or usage(),
        stop_reason=stop_reason,
    )
    return SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=make_async_stream(chunks))
        )
    )
