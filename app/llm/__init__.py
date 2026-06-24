"""LLM layer: model client, prompt-system helpers, streaming primitives.

This package owns every interaction with the Google Gemini API surface:

  - `gemini_client.client` — the shared `genai.Client` singleton
  - `gemini_client.generate_text` — one-shot system+user → text helper for
    the non-streaming auxiliary call sites
  - `gemini_adapter` — `to_gemini_contents` / `to_gemini_tools` /
    `extract_usage` translate the app's Anthropic-shaped data to the SDK
  - `caching.assemble_chat_system` — joins the system prompt into the single
    `system_instruction` string Gemini takes (no per-block cache markers)
  - `streaming.encode_sse_text` — serializes streamed text into SSE frames

The `app/agents/` and `app/domain/` layers depend on this package; nothing
in here depends on either. That keeps the model-API surface in one place.
"""
from app.llm.caching import assemble_chat_system
from app.llm.gemini_adapter import (
    extract_usage,
    to_gemini_contents,
    to_gemini_tools,
)
from app.llm.gemini_client import client, generate_text
from app.llm.streaming import encode_sse_text

__all__ = [
    "assemble_chat_system",
    "client",
    "encode_sse_text",
    "extract_usage",
    "generate_text",
    "to_gemini_contents",
    "to_gemini_tools",
]
