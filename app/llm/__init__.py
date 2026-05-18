"""LLM layer: model clients, prompt-caching helpers, streaming primitives.

This package owns every interaction with the Anthropic API surface:

  - `anthropic_client.client` — the shared AsyncAnthropic singleton
  - `caching.assemble_chat_system` — builds a `list[TextBlockParam]` system
    payload with cache_control markers on the stable prefix
  - `streaming.encode_sse_text` — serializes streamed text into SSE frames
  - `models.PROMPT_CACHING_BETA` — the beta feature flag name to enable
    prompt caching on `client.beta.messages.stream`

The `app/agents/` and `app/domain/` layers depend on this package; nothing
in here depends on either. That keeps the model-API surface in one place.
"""
from app.llm.anthropic_client import client
from app.llm.caching import (
    EPHEMERAL,
    assemble_chat_system,
    cached_text,
    plain_text,
)
from app.llm.models import PROMPT_CACHING_BETA
from app.llm.streaming import encode_sse_text

__all__ = [
    "EPHEMERAL",
    "PROMPT_CACHING_BETA",
    "assemble_chat_system",
    "cached_text",
    "client",
    "encode_sse_text",
    "plain_text",
]
