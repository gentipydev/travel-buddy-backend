"""Conversations domain: persistence, summarization, persona refresh.

Public surface:
  - `get_or_create_conversation`, `append_messages`, `build_history` —
    the CRUD + history-rendering primitives used by the chat router.
  - `maybe_summarize`, `generate_title` — Claude-driven housekeeping
    called after each assistant turn.
  - `refresh_travel_persona` — fire-and-forget background task that
    enriches `UserPreferences.travel_persona` from recent summaries.
"""
from app.domain.conversations.persona import (
    refresh_travel_persona,
)
from app.domain.conversations.repository import (
    append_messages,
    build_history,
    get_or_create_conversation,
)
from app.domain.conversations.summarization import (
    generate_title,
    maybe_summarize,
)

__all__ = [
    "append_messages",
    "build_history",
    "generate_title",
    "get_or_create_conversation",
    "maybe_summarize",
    "refresh_travel_persona",
]
