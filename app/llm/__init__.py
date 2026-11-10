"""LLM layer: model client, prompt-caching helpers, streaming primitives.

This package owns every interaction with the Anthropic API surface:

  - `anthropic_client.client` — the shared `AsyncAnthropic` singleton
  - `anthropic_client.generate_message` — one-shot system+user → `Message`
    for the call sites that need `usage` back (the persona endpoint)
  - `anthropic_client.generate_text` — the text-only wrapper the auxiliary
    (non-streaming) call sites share
  - `anthropic_client.extract_usage` — `Message.usage` → the app's four-field
    token accounting `(input, output, cache_creation, cache_read)`
  - `anthropic_client.classify_llm_error` — collapses the named API failure
    modes to stable prefixes for logs and the SSE error frame
  - `caching.assemble_chat_system` — builds a `list[TextBlock]` system payload
    with a `cache_control` marker on the stable prefix
  - `streaming.encode_sse_text` — serializes streamed text into SSE frames

The `app/agents/` and `app/domain/` layers depend on this package; nothing
in here depends on either. That keeps the model-API surface in one place.
"""
from app.llm.anthropic_client import (
    ANTHROPIC_AUTH_FAILED,
    ANTHROPIC_OVERLOADED,
    ANTHROPIC_RATE_LIMITED,
    ANTHROPIC_UNAVAILABLE,
    classify_llm_error,
    client,
    extract_usage,
    generate_message,
    generate_text,
    message_text,
)
from app.llm.caching import (
    EPHEMERAL,
    assemble_chat_system,
    cached_text,
    plain_text,
)
from app.llm.streaming import encode_sse_text

__all__ = [
    "ANTHROPIC_AUTH_FAILED",
    "ANTHROPIC_OVERLOADED",
    "ANTHROPIC_RATE_LIMITED",
    "ANTHROPIC_UNAVAILABLE",
    "EPHEMERAL",
    "assemble_chat_system",
    "cached_text",
    "classify_llm_error",
    "client",
    "encode_sse_text",
    "extract_usage",
    "generate_message",
    "generate_text",
    "message_text",
    "plain_text",
]
