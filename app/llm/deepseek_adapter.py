"""Adapters between the app's Anthropic-shaped data and the OpenAI/DeepSeek API.

The DeepSeek fallback speaks the OpenAI Chat Completions wire shape. The rest
of the app speaks the original Anthropic-flavored shapes — message dicts with
`{"role": "user"/"assistant", "content": str | [blocks]}`, tool definitions
with a JSON-Schema `input_schema`, and the four-field token accounting. This
module converts at the boundary, the OpenAI-side mirror of `gemini_adapter`.

Three concerns:
  - `to_openai_messages` — system string + message dicts → OpenAI `messages`.
    DeepSeek is TEXT-ONLY, so image blocks are DROPPED (the text still goes
    through) and the caller is told via the returned `images_dropped` flag.
  - `to_openai_tools` — `[{name, description, input_schema}]` →
    `[{"type": "function", "function": {...}}]`. OpenAI takes raw JSON Schema
    as `parameters`, so `input_schema` passes through unchanged.
  - `extract_usage` — OpenAI/DeepSeek `usage` → `(input, output,
    cache_creation, cache_read)` with the SAME semantics `gemini_adapter`
    produces (input EXCLUDES cached tokens; no separate cache-creation
    billing → always 0), so `compute_cost` / the usage ledger are unchanged.
"""
from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool schemas.
# ---------------------------------------------------------------------------


def to_openai_tools(tools: list[dict]) -> list[dict]:
    """Translate Anthropic-style tool dicts into OpenAI `tools`.

    Each input dict is `{"name", "description", "input_schema": {...}}`. The
    OpenAI function-calling format nests them under `function` and takes the
    JSON Schema verbatim as `parameters` — no per-keyword conversion needed
    (unlike Gemini's `types.Schema`). Returns `[]` for no tools.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema") or {"type": "object"},
            },
        }
        for t in tools
    ]


# ---------------------------------------------------------------------------
# Messages.
# ---------------------------------------------------------------------------


def _text_from_content(content: Any) -> tuple[str, bool]:
    """Flatten one message's `content` (str or list of Anthropic blocks) into
    a single text string for the OpenAI API. Returns `(text, dropped_image)`
    — `dropped_image` is True when an image block was present and skipped
    (DeepSeek is text-only)."""
    if isinstance(content, str):
        return content, False
    if not isinstance(content, list):
        return "", False
    parts: list[str] = []
    dropped = False
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text" and block.get("text"):
            parts.append(block["text"])
        elif btype == "image":
            dropped = True
    return "\n".join(parts), dropped


def to_openai_messages(
    system: str, messages: list[dict]
) -> tuple[list[dict], bool]:
    """Translate the assembled system string + Anthropic-format `messages`
    into OpenAI `messages`, returning `(messages, images_dropped)`.

    Role mapping: `assistant` → `assistant`, everything else → `user`. The
    system prompt becomes the leading `{"role": "system"}` message. Image
    blocks are dropped (text-only model) and surfaced via `images_dropped`;
    messages that flatten to empty text are skipped so we never send an empty
    turn. A warning is logged once when any image is dropped.
    """
    out: list[dict] = []
    if system:
        out.append({"role": "system", "content": system})
    images_dropped = False
    for m in messages:
        role = "assistant" if m.get("role") == "assistant" else "user"
        text, dropped = _text_from_content(m.get("content"))
        images_dropped = images_dropped or dropped
        if text:
            out.append({"role": role, "content": text})
    if images_dropped:
        logger.warning(
            "deepseek_adapter: dropped image part(s) — DeepSeek fallback is "
            "text-only; proceeding without the image(s)"
        )
    return out, images_dropped


# ---------------------------------------------------------------------------
# Usage / token accounting.
# ---------------------------------------------------------------------------


def extract_usage(usage: Any) -> tuple[int, int, int, int]:
    """Map an OpenAI/DeepSeek `usage` object to the app's four-field
    accounting: `(input_tokens, output_tokens, cache_creation_input_tokens,
    cache_read_input_tokens)`.

    DeepSeek reports `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens`
    (the miss portion is the full-rate input; the hit portion is the cheap
    cached read) plus `completion_tokens`. Falls back to OpenAI's
    `prompt_tokens` / `prompt_tokens_details.cached_tokens` when the DeepSeek
    fields are absent. No separate cache-creation billing → always 0. Returns
    all zeros when usage is missing — telemetry must never crash.
    """
    if usage is None:
        return (0, 0, 0, 0)
    output_tokens = getattr(usage, "completion_tokens", None) or 0
    miss = getattr(usage, "prompt_cache_miss_tokens", None)
    hit = getattr(usage, "prompt_cache_hit_tokens", None)
    if miss is not None or hit is not None:
        input_tokens = miss or 0
        cache_read = hit or 0
    else:
        # OpenAI-shape fallback: prompt_tokens INCLUDES cached, mirror Gemini.
        prompt = getattr(usage, "prompt_tokens", None) or 0
        details = getattr(usage, "prompt_tokens_details", None)
        cached = getattr(details, "cached_tokens", None) or 0 if details else 0
        input_tokens = max(0, prompt - cached)
        cache_read = cached
    return (input_tokens, output_tokens, 0, cache_read)
