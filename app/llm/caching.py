"""Anthropic prompt-caching helpers.

Prompt caching lets us re-send a stable system prompt without paying for
the input tokens every turn. The Anthropic API marks a cacheable
"breakpoint" with `cache_control: {"type": "ephemeral"}` on a block; on
the next request within ~5 minutes that breakpoint is reused, and only
the suffix counts as billable input.

Single highest-leverage win on the chat path: the ~3-4k tokens of base
prompt + addenda + preferences are stable across turns within a session.
Refine turns carry a large but per-turn-volatile `trip_json`; we cache
only the prefix and leave `trip_json` uncached.

`assemble_chat_system` returns a `list[dict]` compatible with the
Anthropic SDK's `PromptCachingBetaTextBlockParam`. The chat agent passes
it as `system=` to `client.beta.prompt_caching.messages.stream(...)`; the
SDK injects the `anthropic-beta: prompt-caching-2024-07-31` header.
"""
from typing import Literal, TypedDict


class CacheControl(TypedDict):
    type: Literal["ephemeral"]


class TextBlock(TypedDict, total=False):
    type: Literal["text"]
    text: str
    cache_control: CacheControl


EPHEMERAL: CacheControl = {"type": "ephemeral"}

# Anthropic caps cache_control markers at 4 per request. We use 1 today
# (the chat prefix); the limit is documented here so future call sites
# don't accidentally exceed it.
_MAX_BREAKPOINTS = 4


def cached_text(text: str) -> TextBlock:
    """A text block marked as a cache breakpoint."""
    return {"type": "text", "text": text, "cache_control": EPHEMERAL}


def plain_text(text: str) -> TextBlock:
    """A text block that is NOT a cache breakpoint."""
    return {"type": "text", "text": text}


def assemble_chat_system(
    *,
    cached_prefix: str,
    volatile_suffix: str = "",
) -> list[TextBlock]:
    """Build a 2-block system payload: a cached prefix + an optional
    uncached suffix.

    Pass an empty `volatile_suffix` to cache the whole system. The chat
    agent uses the suffix for the refine addendum (carries trip_json that
    varies per turn) and for retrieved RAG context (varies per query).
    """
    blocks: list[TextBlock] = [cached_text(cached_prefix)]
    if volatile_suffix:
        blocks.append(plain_text(volatile_suffix))
    return blocks
