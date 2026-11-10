"""Anthropic prompt-caching helpers.

Prompt caching lets us re-send a stable system prompt without paying for the
input tokens every turn. The API marks a cacheable "breakpoint" with
`cache_control: {"type": "ephemeral"}` on a block; on the next request within
~5 minutes that breakpoint is reused, and only the suffix counts as billable
input. Caching is GA — it needs no beta header and no special client path:
pass these blocks as `system=` to `client.messages.stream(...)`.

Single highest-leverage win on the chat path: the base prompt + addenda +
preferences are stable across turns within a session. Refine turns carry a
large but per-turn-volatile `trip_json`; we cache only the prefix and leave
`trip_json` uncached.

CAVEAT — on `claude-haiku-4-5` the chat prefix is currently just BELOW the
minimum cacheable size, so plain chat turns do not cache. Measured against
the live API on 2026-08-25 (not estimated):

    system prompt (no preferences)          2504 tokens
    ITINERARY_TOOL                          1738 tokens
    ------------------------------------------------
    non-refine cacheable prefix             4242 tokens  -> does NOT cache
    + REFINE_TOOLS (refine turns only)      2462 tokens
    ------------------------------------------------
    refine cacheable prefix                 6704 tokens  -> caches

A bracketing probe on this exact prompt put the effective threshold between
4244 (no cache) and 4546 (caches) — so plain turns miss by roughly 300
tokens, while refine turns clear it comfortably. Note the published minimum
for this model is quoted as 4096; the observed cutoff is slightly higher, so
trust the measurement over the number.

The marker below is placed correctly on every turn regardless, and the wiring
is verified (a padded control prompt caches and re-reads exactly as expected).
So this starts paying off the moment the prefix grows past the cutoff — adding
a preferences block, extending the tool schema — or the model changes (Sonnet
5's minimum is 1024). Until then, expect `cache_read_input_tokens == 0` on
plain turns and non-zero on refine turns. Do not chase the zero as a defect;
re-measure before assuming it changed. See docs/ai-patterns.md.
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
    `app/llm/token_budget.py` only ever trims the volatile suffix, for the
    same reason: trimming the prefix would invalidate the cache.
    """
    blocks: list[TextBlock] = [cached_text(cached_prefix)]
    if volatile_suffix:
        blocks.append(plain_text(volatile_suffix))
    return blocks
