"""Compose Claude system prompts from the registry for /chat/stream.

Two public callables:

  - `build_system_payload(prefs, trip_context)` — returns a
    `(cached_prefix, volatile_suffix)` tuple. The prefix is stable across
    turns within a session and is the cache breakpoint passed to Claude's
    prompt-caching API; the suffix carries the refine addendum (per-turn
    `trip_json`) and is concatenated uncached. The /chat/stream agent
    composes the two halves via `app/llm/caching.assemble_chat_system`.
  - `build_system_prompt(prefs, trip_context)` — legacy single-string
    output. Calls `build_system_payload` and concatenates so the byte
    representation matches the pre-migration `app/prompts.py` output.

The chat pipeline appends retrieved RAG chunks to the volatile suffix
(chunks differ per query and would cache-bust the prefix).
"""
import json

from app.db.models import UserPreferences
from app.prompts.registry import get


def build_system_payload(
    preferences: UserPreferences | None,
    trip_context: dict | None = None,
) -> tuple[str, str]:
    """Return `(cached_prefix, volatile_suffix)` for the chat system payload.

    `cached_prefix` is BASE + inference/override addendum + itinerary-tool
    addendum (and the preferences block when prefs are set). This is the
    ~3-4k tokens that don't change turn-to-turn.

    `volatile_suffix` is the refine addendum rendered with `trip_json`
    when refine mode is active, else "".
    """
    base = get("chat.base").text
    inference = get("chat.inference_addendum").text
    override = get("chat.override_reminder").text
    itinerary = get("chat.itinerary_tool_addendum").text
    refine_template = get("chat.refine_addendum").text

    # Case 1: no preferences row at all.
    # Case 2: row exists but nothing meaningful is set.
    if preferences is None or not preferences.has_preferences():
        prefix = base + inference + itinerary
    else:
        # Case 3: preferences exist — inject as defaults with override reminder.
        prefix = (
            f"{base}\n\n"
            "User preferences (defaults only — see override rule below):\n"
            f"{preferences.to_prompt()}"
            f"{override}"
            f"{itinerary}"
        )

    if trip_context is not None:
        # `indent=2` adds a few hundred tokens of structure that materially
        # improves Claude's adherence to the trip shape on full re-emit. The
        # gating phrasing below the JSON is load-bearing — it's the entire
        # commit-vs-explore confirmation surface (no client-side dialog).
        suffix = refine_template.format(
            trip_json=json.dumps(trip_context, indent=2)
        )
    else:
        suffix = ""

    return prefix, suffix


def build_system_prompt(
    preferences: UserPreferences | None,
    trip_context: dict | None = None,
) -> str:
    """Legacy single-string assembler.

    Byte-equivalent to the pre-migration `app/prompts.py:build_system_prompt`.
    New code should prefer `build_system_payload` so the cached prefix and
    volatile suffix stay distinguishable for the caching layer.
    """
    prefix, suffix = build_system_payload(preferences, trip_context)
    return prefix + suffix
